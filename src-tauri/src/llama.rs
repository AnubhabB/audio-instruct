use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use anyhow::Result;
use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
};
use tokenizers::Tokenizer;

use crate::utils::{device, hf_download};

const MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct.Q8_0.gguf";
const MODEL_REPO: &str = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF";

const TOKENIZER_REPO: &str = "unsloth/llama-3-8b";
const LOCAL_MODEL_TOK: &str = "llama3-tokenizer.json";

const TEMPERATURE: f64 = 0.8;
const TOP_P: f64 = 0.95;
const TOP_K: usize = 40;

const MAX_NEW_TOKENS: usize = 2048;

/// A struct to maintain a initialized Llama quantized `gguf` model and associated methods
pub struct LlamaWrap {
    device: Device,
    model: Arc<Mutex<ModelWeights>>,
    tokenizer: Tokenizer,
    sampler: Arc<Mutex<LogitsProcessor>>,
    stop_tokens: [u32; 2],
}

impl LlamaWrap {
    /// Initializer for new llama manager
    pub fn new(dir: &Path) -> Result<Self> {
        let device = device()?;

        let model_path = Self::model_path(dir)?;
        let (model, tokenizer) = Self::load_model(model_path.as_path(), &device)?;
        let stop_tokens = [
            tokenizer.token_to_id("<|eot_id|>").unwrap(),
            tokenizer.token_to_id("<|end_of_text|>").unwrap(),
        ];

        let sampler = Arc::new(Mutex::new(LogitsProcessor::from_sampling(
            42,
            Sampling::TopKThenTopP {
                k: TOP_K,
                p: TOP_P,
                temperature: TEMPERATURE,
            },
        )));

        info!("Llama ready!");
        Ok(Self {
            device,
            model,
            tokenizer,
            sampler,
            stop_tokens,
        })
    }

    fn model_path(base_dir: &Path) -> Result<PathBuf> {
        let model_path = base_dir;

        // The file doesn't exist, lets download it
        if !model_path.join(MODEL_FILE).is_file() {
            hf_download(base_dir, MODEL_REPO, MODEL_FILE, None)?;
        }

        // Download the tokenizer.json, and rename it, its a separate repo
        if !model_path.join(LOCAL_MODEL_TOK).is_file() {
            hf_download(
                base_dir,
                TOKENIZER_REPO,
                "tokenizer.json",
                Some(LOCAL_MODEL_TOK),
            )?;
        }

        Ok(model_path.to_path_buf())
    }

    fn load_model(
        model_dir: &Path,
        device: &Device,
    ) -> Result<(Arc<Mutex<ModelWeights>>, Tokenizer)> {
        let model_file = model_dir.join(MODEL_FILE);
        let tok_file = model_dir.join(LOCAL_MODEL_TOK);

        info!("Loading gguf model @{:?}", model_file);

        let mut file = std::fs::File::open(model_file)?;
        // reading the params from file
        let model = gguf_file::Content::read(&mut file)?;

        let model = Arc::new(Mutex::new(ModelWeights::from_gguf(
            model, &mut file, device,
        )?));

        info!("Loading tokenizer @{:?}", tok_file);
        let tokenizer = Tokenizer::from_file(tok_file).unwrap();

        Ok((model, tokenizer))
    }

    /// Helper function to convert incoming `command` to templated prompt
    /// Prompt template: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202
    pub fn preproc(txt: &str) -> String {
        format!(
            "<|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable, efficient, intelligent and direct AI assistant. Provide concise answers, focusing on the key information needed. Respond only with the answer to the instruction based on the given data. Do not add any additional text, introduction, context or explanation. If you are unsure about an answer, truthfully return \"Not Known\".<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            txt
        )
    }

    // returns the (generated text, number of tokens generated, duration)
    pub fn infer(&self, instruct: &str) -> Result<(String, usize, std::time::Duration)> {
        let prompt = Self::preproc(instruct);
        let prompttokens = match self.tokenizer.encode(prompt, true) {
            Ok(t) => t,
            Err(e) => {
                error!("infer: error tokenizing prompt tokens: {e:?}");
                anyhow::bail!("error tokenizing prompt");
            }
        };

        let mut model = match self.model.lock() {
            Ok(m) => m,
            Err(e) => {
                error!("infer: acquiring model mutex lock: {e:?}");
                anyhow::bail!("error acquiring model");
            }
        };

        let mut sampler = match self.sampler.lock() {
            Ok(s) => s,
            Err(e) => {
                error!("infer: acquiring model mutex lock: {e:?}");
                anyhow::bail!("error acquiring model");
            }
        };

        let mut all_tokens = vec![];
        let start_prompt_processing = std::time::Instant::now();

        let mut input = Tensor::new(prompttokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let mut logits = model.forward(&input, 0)?;
        let mut next = sampler.sample(&logits.squeeze(0)?)?;

        all_tokens.push(next);

        for i in prompttokens.len()..MAX_NEW_TOKENS {
            input = Tensor::new(&[next], &self.device)?.unsqueeze(0)?;

            logits = model.forward(&input, i)?;
            next = sampler.sample(&logits.squeeze(0)?)?;

            if self.stop_tokens.contains(&next) {
                break;
            }

            all_tokens.push(next);
        }

        let tk = match self.tokenizer.decode(&all_tokens[..], false) {
            Ok(t) => t,
            Err(e) => {
                error!("Error generating tokens: {e:?}");
                anyhow::bail!("Error generating tokens")
            }
        };

        Ok((
            tk,
            all_tokens.len(),
            std::time::Instant::now().duration_since(start_prompt_processing),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::LlamaWrap;

    #[test]
    fn llama_infer() -> anyhow::Result<()> {
        pretty_env_logger::init();

        let dir = Path::new("/Users/anubhab/Library/Application Support/audio-instruct.llm");
        let llama = LlamaWrap::new(dir)?;

        let inf = llama.infer("Who is Steve Wozniak?")?;

        info!("{inf:?}");
        Ok(())
    }
}
