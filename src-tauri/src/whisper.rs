use std::path::{Path, PathBuf};

use anyhow::{anyhow, Result};
use candle_core::Device;
use candle_nn::VarBuilder;
use candle_transformers::models::whisper::{model::Whisper, Config};
use tokenizers::Tokenizer;

use crate::utils::{device, hf_download};


const MODEL_REPO: &str = "distil-whisper/distil-large-v3";

/// A struct to hold `distil-whisper` object and associated methods
pub struct WhisperWrap {
    mel_filters: Vec<f32>,
    model: Whisper,
    tokenizer: Tokenizer
}

impl WhisperWrap {
    pub fn new(dir: &Path) -> Result<Self> {
        let device = device()?;

        let model_path = Self::model_path(dir)?;
        let (model, mel_filters, tokenizer) = Self::load_model(model_path.as_path(), &device)?;

        info!("Whisper ready!");
        Ok(
            Self { mel_filters, model, tokenizer }
        )
    }

    // checks if the model file(s) exists or not and downloads it
    // Unlike a `gguf` model, we'll need 33 files for our model to work
    // the model weights will be in `model.safetensors`
    // the tokenizer.json file for the vocab
    // config.json for the model config
    fn model_path(base_dir: &Path) -> Result<PathBuf> {
        // Check if tokenizer.json exists, or create
        if !base_dir.join("tokenizer.json").is_file() {
            hf_download(base_dir, MODEL_REPO, "tokenizer.json")?;
        }

        // check if config.json exists else download
        if !base_dir.join("config.json").is_file() {
            hf_download(base_dir, MODEL_REPO, "config.json")?;
        }

        // finally, check if model.safetensors exist or download
        // this is a large file, and might take a while
        if !base_dir.join("model.safetensors").is_file() {
            hf_download(base_dir, MODEL_REPO, "model.safetensors")?;
        }

        Ok(base_dir.to_path_buf())
    }

    fn load_model(model_dir: &Path, device: &Device) -> Result<(Whisper, Vec<f32>, Tokenizer)> {
        info!("Loading whisper");

        let tokenizer = match Tokenizer::from_file(model_dir.join("tokenizer.json")) {
            Ok(t) => t,
            Err(e) => {
                error!("Error loading tokenizer: {e:?}");
                return Err(anyhow!("{e:?}"));
            }
        };

        let config: Config = serde_json::from_str(&std::fs::read_to_string(model_dir.join("config.json"))?)?;

        let mel = match config.num_mel_bins {
            80 => include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters.bytes")).as_slice(),
            128 => include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/melfilters128.bytes")).as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(mel, &mut mel_filters);


        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_dir.join("model.safetensors")], candle_core::DType::F16, device)? };
        let model = Whisper::load(&vb, config)?;

        Ok((
            model,
            mel_filters,
            tokenizer
        ))
    }
}