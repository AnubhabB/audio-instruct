use std::path::Path;

use anyhow::Result;
use candle_core::Device;
use hf_hub::api::sync::ApiBuilder;

/// a helper function to download a file and move to a specific path from `huggingface hub`
pub fn hf_download(dir: &Path, repo: &str, file: &str, rename: Option<&str>) -> Result<()> {
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
    std::fs::rename(path, dir.join(rename.map_or(file, |rname| rname)))?;

    // We'll also delete the download directory created by `hf` -- this adds no other value than just cleaning up our data directory
    let toclean = dir.join(format!(
        "models--{}",
        repo.split("/").collect::<Vec<_>>().join("--")
    ));
    std::fs::remove_dir_all(toclean)?;

    Ok(())
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

/// A helper function to convert incomig &[u8] to Vec<f32>
pub fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}
