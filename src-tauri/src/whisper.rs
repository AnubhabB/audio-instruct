use core::f64;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{anyhow, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::whisper::{
    audio::pcm_to_mel, model::Whisper, Config, DTYPE, EOT_TOKEN, LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD, NO_TIMESTAMPS_TOKEN, N_FRAMES, SOT_TOKEN, TEMPERATURES, TRANSCRIBE_TOKEN,
};
use rand::prelude::Distribution;
use tokenizers::Tokenizer;

use crate::utils::{device, hf_download};

const MODEL_REPO: &str = "distil-whisper/distil-large-v3";

const LOCAL_MODEL_CFG: &str = "whisper-config.json";
const LOCAL_MODEL_TOK: &str = "whisper-tokenizer.json";
const LOCAL_MODEL_MODEL: &str = "whisper-model.safetensors";

/// Enabling inference for English only
const LANGUAGE: &str = "en";

#[derive(Debug)]
struct DecodingResult {
    tokens: Vec<u32>,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
}

/// A struct to hold `distil-whisper` object and associated methods
pub struct WhisperWrap {
    device: Device,
    data: Arc<Mutex<Vec<f32>>>,
    config: Config,
    mel_filters: Vec<f32>,
    model: Arc<Mutex<Whisper>>,
    tokenizer: Tokenizer,
    default_tokens: DefaultTokens,
}

pub struct DefaultTokens {
    // start of transcript token
    sot: u32,
    // end of transcript
    eot: u32,
    // token representing language
    lang: u32,
    // token representing task - for us it's transcribe
    transcribe: u32,
    // token for no-timestamp - we don't need timestamp
    no_ts: u32,
    // token representing no speech
    no_speech: u32,
}

impl DefaultTokens {
    pub fn init(tk: &Tokenizer) -> Self {
        Self {
            sot: tk.token_to_id(SOT_TOKEN).unwrap(),
            eot: tk.token_to_id(EOT_TOKEN).unwrap(),
            lang: tk.token_to_id(format!("<|{LANGUAGE}|>").as_str()).unwrap(),
            transcribe: tk.token_to_id(TRANSCRIBE_TOKEN).unwrap(),
            no_ts: tk.token_to_id(NO_TIMESTAMPS_TOKEN).unwrap(),
            no_speech: tk.token_to_id("<|nospeech|>").unwrap(),
        }
    }
}

impl WhisperWrap {
    pub fn new(dir: &Path) -> Result<Self> {
        let device = device()?;

        let model_path = Self::model_path(dir)?;
        let (model, mel_filters, tokenizer, config) =
            Self::load_model(model_path.as_path(), &device)?;

        info!("Whisper ready!");
        Ok(Self {
            device,
            config,
            data: Arc::new(Mutex::new(Vec::new())),
            default_tokens: DefaultTokens::init(&tokenizer),
            mel_filters,
            model,
            tokenizer,
        })
    }

    // checks if the model file(s) exists or not and downloads it
    // Unlike a `gguf` model, we'll need 33 files for our model to work
    // the model weights will be in `model.safetensors`
    // the tokenizer.json file for the vocab
    // config.json for the model config
    fn model_path(base_dir: &Path) -> Result<PathBuf> {
        // Check if tokenizer.json exists, or create
        if !base_dir.join(LOCAL_MODEL_TOK).is_file() {
            hf_download(
                base_dir,
                MODEL_REPO,
                "tokenizer.json",
                Some(LOCAL_MODEL_TOK),
            )?;
        }

        // check if config.json exists else download
        if !base_dir.join(LOCAL_MODEL_CFG).is_file() {
            hf_download(base_dir, MODEL_REPO, "config.json", Some(LOCAL_MODEL_CFG))?;
        }

        // finally, check if model.safetensors exist or download
        // this is a large file, and might take a while
        if !base_dir.join(LOCAL_MODEL_MODEL).is_file() {
            hf_download(
                base_dir,
                MODEL_REPO,
                "model.safetensors",
                Some(LOCAL_MODEL_MODEL),
            )?;
        }

        Ok(base_dir.to_path_buf())
    }

    fn load_model(
        model_dir: &Path,
        device: &Device,
    ) -> Result<(Arc<Mutex<Whisper>>, Vec<f32>, Tokenizer, Config)> {
        info!("Loading whisper");

        let tokenizer = match Tokenizer::from_file(model_dir.join(LOCAL_MODEL_TOK)) {
            Ok(t) => t,
            Err(e) => {
                error!("Error loading tokenizer: {e:?}");
                return Err(anyhow!("{e:?}"));
            }
        };

        let config: Config =
            serde_json::from_str(&std::fs::read_to_string(model_dir.join(LOCAL_MODEL_CFG))?)?;

        let mel = match &config.num_mel_bins {
            80 => {
                include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters.bytes")).as_slice()
            }
            128 => include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters128.bytes"))
                .as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel, &mut mel_filters);

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join(LOCAL_MODEL_MODEL)],
                DTYPE,
                device,
            )?
        };
        let model = Arc::new(Mutex::new(Whisper::load(&vb, config.clone())?));

        Ok((model, mel_filters, tokenizer, config))
    }

    fn preproc(&self) -> Result<Tensor> {
        let data = match self.data.lock() {
            Ok(mut d) => {
                if d.len() < 4096 * 4 {
                    anyhow::bail!("Not enough audio data in buffer!");
                }
                let d = d.drain(..).collect::<Vec<_>>();

                d
            }
            Err(e) => {
                error!("error acquiring data lock: {e:?}");
                anyhow::bail!("Not enough audio data in buffer!");
            }
        };

        let mel = pcm_to_mel(&self.config, &data[..], &self.mel_filters[..]);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (
                1,
                self.config.num_mel_bins,
                mel_len / self.config.num_mel_bins,
            ),
            &self.device,
        )?;

        Ok(mel)
    }

    fn preproc_decode(&self) -> Vec<u32> {
        vec![
            self.default_tokens.sot,
            self.default_tokens.lang,
            self.default_tokens.transcribe,
            self.default_tokens.no_ts,
        ]
    }

    fn decode_segment(&self, model: &mut Whisper, segment: &Tensor) -> Result<DecodingResult> {
        for (i, &t) in TEMPERATURES.iter().enumerate() {
            let decoded = self.decode(model, segment, t);
            if i == TEMPERATURES.len() - 1 {
                return decoded;
            }

            match decoded {
                Ok(decoded) => {
                    if decoded.avg_logprob >= LOGPROB_THRESHOLD
                        || decoded.no_speech_prob > NO_SPEECH_THRESHOLD
                    {
                        return Ok(decoded);
                    }

                    warn!(
                        "No data for decoding @ temperature: {}",
                        decoded.temperature
                    );
                }
                Err(e) => {
                    warn!("Error decoding @ temperature: {t}: {e:?}");
                }
            }
        }

        unreachable!()
    }

    fn decode(&self, model: &mut Whisper, segment: &Tensor, temp: f64) -> Result<DecodingResult> {
        let mut rng = rand::thread_rng();

        let features = model.encoder.forward(segment, true)?;
        let mut tokens = self.preproc_decode();

        let mut no_speech_prob = f64::MAX;
        let mut sum_log_p = 0.;

        for i in 0..self.config.max_target_positions {
            let tensor = Tensor::new(&tokens[..], &self.device)?;
            let dec = model
                .decoder
                .forward(&tensor.unsqueeze(0)?, &features, i == 0)?;

            // Extract the no speech probability on the first iteration by looking at the first
            // token logits and the probability for the according token.
            if i == 0 {
                let logits = model.decoder.final_linear(&dec.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.default_tokens.no_speech as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = dec.dims3()?;
            let logits = model
                .decoder
                .final_linear(&dec.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;

            let next_token = if temp > 0. {
                let prs = softmax(&(&logits / temp)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = rand::distributions::WeightedIndex::new(&logits_v)?;
                distr.sample(&mut rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);

            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;
            if next_token == self.default_tokens.eot
                || tokens.len() > self.config.max_target_positions
            {
                break;
            }
            sum_log_p += prob.ln();
        }

        Ok(DecodingResult {
            avg_logprob: sum_log_p / tokens.len() as f64,
            tokens,
            no_speech_prob,
            temperature: temp,
        })
    }

    /// Accepts an incoming chunk of data and appends it to our `data` field of the struct
    pub fn chunk(&self, chunk: Vec<f32>) {
        let mut c = chunk;
        let mut chunk = self.data.lock().unwrap();

        chunk.append(&mut c);
    }

    /// Runs transcription
    pub fn infer(&self) -> Result<(String, usize, std::time::Duration)> {
        let mels = self.preproc()?;

        let mut model = match self.model.lock() {
            Ok(m) => m,
            Err(e) => {
                error!("infer: error acquiring model lock: {e:?}");
                anyhow::bail!("error during inference");
            }
        };

        let (_, _, content_frames) = mels.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];

        // newline tokens after each segment
        let nltokens = self
            .tokenizer
            .encode("\n", false)
            .unwrap()
            .get_ids()
            .to_vec();

        let mut total_dur = Duration::from_millis(0);
        let mut total_tokens = 0;

        while seek < content_frames {
            let start = std::time::Instant::now();

            let segment_size = usize::min(content_frames - seek, N_FRAMES);
            let mel_segment = mels.narrow(2, seek, segment_size)?;

            let mut decoded = self.decode_segment(&mut model, &mel_segment)?;
            seek += segment_size;

            total_dur += std::time::Instant::now() - start;
            total_tokens += decoded.tokens.len();

            if decoded.no_speech_prob > NO_SPEECH_THRESHOLD
                && decoded.avg_logprob < LOGPROB_THRESHOLD
            {
                println!("no speech detected, skipping {seek} {decoded:?}");
                continue;
            }

            segments.append(&mut decoded.tokens);
            // adding newline tokens after each segment
            nltokens.iter().for_each(|&t| {
                segments.push(t);
            });
        }

        // Let us now create the final text output
        let instruct = self
            .tokenizer
            .decode(&segments, true)
            .map_err(|_| anyhow!("error creating text from tokens"))?;

        Ok((instruct, total_tokens, total_dur))
    }
}
