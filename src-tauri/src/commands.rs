use std::sync::Arc;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tauri::ipc;

use crate::{instruct::Instruct, utils::bytes_to_f32};

/// A struct to represent the incoming request
/// For `text` inference it would contain the instruction itself,
/// But for `audio` instructions we'll just need to indicate that we are looking for the audio recorded to generate inference since audio data is sent in chunks and
/// already maintained in the app state
#[derive(Debug, Deserialize)]
pub struct Command {
    text: Option<String>,
    audio: Option<bool>
}

/// Enum to maintain what kind of instruction this is
pub enum Mode {
    Text(String),
    Audio
}

impl Command {
    pub fn mode(&self) -> Result<Mode> {
        if let Some(t) = self.text.as_ref() {
            Ok(Mode::Text(t.to_owned()))
        } else if self.audio.map_or(false, |d| d) {
            Ok(Mode::Audio)
        } else {
            anyhow::bail!("not a valid command")
        }
    }
}

/// A struct to hold the response data and some stats or metadata required to show the inference
#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    text: String,
    meta: Meta
}

/// A struct to hold some metadata and additional information about the QA/ Response/ Instruction etc.
#[derive(Debug, Serialize, Deserialize)]
pub struct Meta {
    // number of tokens generated
    n_tokens: u32,
    // number of seconds elapsed
    n_secs: u64
}

impl Response {
    pub fn new(txt: &str, n_tokens: u32, n_secs: u64) -> Self {
        Self {
            text: txt.to_string(),
            meta: Meta { n_secs, n_tokens }
        }
    }
}

/// A command to accept incoming `instruction` and respond with the `inference`
#[tauri::command]
pub fn ask(
    app: tauri::State<'_, Arc<Instruct>>,
    cmd: Command
) -> Result<Response, &'static str> {
    let command = match cmd.mode() {
        Ok(c) => c,
        Err(e) => {
            error!("ask: invalid incoming command: {e:?}");
            return Err("invalid command")
        }
    };
    
    let res = match command {
        Mode::Text(t) => app.text(&t),
        Mode::Audio => app.audio()
    };

    match res {
        Ok(r) => Ok(r),
        Err(e) => {
            error!("ask: error during inference: {e:?}");
            Err("inference error")
        }
    }
}

/// This tauri command would receive a Vec<f32> which represents a chunk of audio being recorded
/// The chunk will be forwarded through the MPSC channel
#[tauri::command]
pub fn audio_chunk(
    app: tauri::State<'_, Arc<Instruct>>,
    req: ipc::Request<'_>
) -> Result<(), &'static str> {
    if let tauri::ipc::InvokeBody::Raw(data) = req.body() {
        let chunk = bytes_to_f32(&data[..]);
        if let Err(e) = app.send(chunk) {
            error!("audio_chunk: error: {e:?}");
            return Err("invalid chunk")    
        }
    } else {
        return Err("invalid chunk")
    }
    Ok(())
}