use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use candle_core::Device;
use tauri::api::path::data_dir;
use tokenizers::Tokenizer;

use crate::APP_PACKAGE;

/// Helper function to return our app's data directory and create it if it doesn't exist
/// This should basically run just once
pub fn app_data_dir() -> Result<PathBuf> {
    let data_dir = if let Some(d) = data_dir() {
        // Creating an unique name for our app - easier to maintain
        d.join(APP_PACKAGE)
    } else {
        error!("tauri didn't return any data directory");
        return Err(anyhow!("error reading data directory"));
    };

    // if data dir doesn't exist, create it
    create_dir_if_not_exists(&data_dir)?;

    Ok(data_dir)
}

/// helper function to create a directory if it doesn't exist
pub fn create_dir_if_not_exists(p: &Path) -> Result<()> {
    if p.is_dir() {
        return Ok(());
    }

    create_dir_all(p)?;

    Ok(())
}

/// Helper function to convert incoming `command` to templated prompt
/// Prompt template: https://github.com/meta-llama/llama3/blob/main/llama/tokenizer.py#L202
pub fn prompt(txt: &str) -> String {
    format!(
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable, efficient, intelligent and direct AI assistant. Provide concise answers, focusing on the key information needed. Respond only with the answer to the instruction based on the given data. Do not add any additional text, introduction, context or explanation. If you are unsure about an answer, truthfully return \"Not Known\".<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        txt
    )
}


/// A helper function to detect the compute device
pub fn device() -> Result<Device> {
    let dev = if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else {
        Device::Cpu
    };

    Ok(dev)
}


pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}