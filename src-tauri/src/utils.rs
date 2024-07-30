use std::{
    fs::create_dir_all,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use candle_core::Device;
use hf_hub::api::sync::ApiBuilder;
// use tauri::api::path::data_dir;
use tokenizers::Tokenizer;

/// helper function to create a directory if it doesn't exist
pub fn create_dir_if_not_exists(p: &Path) -> Result<()> {
    if p.is_dir() {
        return Ok(());
    }

    create_dir_all(p)?;

    Ok(())
}

/// a helper function to download a file and move to a specific path from `huggingface hub`
pub fn hf_download(dir: &Path, repo: &str, file: &str) -> Result<()> {
    let path = ApiBuilder::new()
            .with_cache_dir(dir.to_path_buf())
            .with_progress(true)
            .build()?
            .model(repo.to_string())
            .get(file)?;

    info!("File downloaded @ {path:?}");
    // The downloaded file path is actually a symlink
    let path = std::fs::canonicalize(&path)?;
    info!("Symlink pointed file: {path:?}");

    // lets move the file to `<app_data_dir>/<file>`, this will ensure that we don't end up downloading the file on the next launch
    // This not required, but just cleaner for me to look at and maintain :)
    std::fs::rename(path, dir.join(file))?;

    // We'll also delete the download directory created by `hf` -- this adds no other value than just cleaning up our data directory
    let toclean = dir.join(format!(
        "models--{}",
        repo.split("/").collect::<Vec<_>>().join("--")
    ));
    std::fs::remove_dir_all(toclean)?;

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
    let dev = Device::new_metal(0)?;

    Ok(dev)
}


pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}