use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use hf_hub::api::sync::ApiBuilder;
// use tauri::api::path::data_dir;
use tokenizers::Tokenizer;

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
    let dev = if cfg!(feature = "metal") {
        Device::new_metal(0)?
    } else if cfg!(feature = "cuda") {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };

    info!("Device: {dev:?}");

    Ok(dev)
}


pub fn token_id(tokenizer: &Tokenizer, token: &str) -> Result<u32> {
    match tokenizer.token_to_id(token) {
        None => anyhow::bail!("no token-id for {token}"),
        Some(id) => Ok(id),
    }
}

/// A helper function to convert incomig &[u8] to Vec<f32>
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}