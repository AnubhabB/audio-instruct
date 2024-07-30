use std::path::{Path, PathBuf};

use anyhow::Result;
use candle_core::{quantized::gguf_file, Device};
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::utils::{device, hf_download};

const MODEL_FILE: &str = "Meta-Llama-3-8B-Instruct.Q8_0.gguf";
const MODEL_REPO: &str = "QuantFactory/Meta-Llama-3-8B-Instruct-GGUF";

/// A struct to maintain a initialized Llama quantized `gguf` model and associated methods
pub struct LlamaWrap {
    model: ModelWeights
}

impl LlamaWrap {
    /// Initializer for new llama manager
    pub fn new(dir: &Path) -> Result<Self> {
        let device = device()?;

        let model_path = Self::model_path(dir)?;
        let model = Self::load_model(model_path.as_path(), &device)?;
        
        info!("Llama ready!");
        Ok(Self { model })
    }

    fn model_path(base_dir: &Path) -> Result<PathBuf> {
        let model_path = base_dir.join(MODEL_FILE);

        // The file doesn't exist, lets download it
        if !model_path.is_file() {
            hf_download(base_dir, MODEL_REPO, MODEL_FILE)?;
        }

        Ok(model_path)
    }

    fn load_model(model_file: &Path, device: &Device) -> Result<ModelWeights> {
        info!("Loading gguf model @{:?}", model_file);

        let mut file = std::fs::File::open(model_file)?;
        // reading the params from file
        let model = gguf_file::Content::read(&mut file)?;

        let model = ModelWeights::from_gguf(model, &mut file, device)?;

        Ok(model)
    }
}