use std::{path::PathBuf, sync::{mpsc::{channel, Receiver, Sender}, Arc}, thread};

use anyhow::Result;

use crate::{llama::LlamaWrap, whisper::WhisperWrap};

#[derive(Debug, Clone, Copy)]
pub enum Event {
    AudioStart,
    AudioEnd,
    AudioData
}

/// A struct to maintain our app state
pub struct Instruct {
    /// Holds the app data directory
    datadir: PathBuf,
    /// holds the instantiated `Llama` quantized `gguf` model and associated methods
    llama: LlamaWrap,
    /// holds the `distil-whisper` model and the associated methods
    whisper: WhisperWrap,
    /// a channel for triggering Instruct methods through events
    send: Sender<Event>,
}

impl Instruct {
    pub fn new(datadir: PathBuf) -> Result<Arc<Self>> {
        let llama = LlamaWrap::new(datadir.as_path())?;
        let whisper = WhisperWrap::new(datadir.as_path())?;

        let (send, recv) = channel();

        let app = Arc::new(Self {
            llama, whisper, send: send.clone(), datadir
        });

        // spawn a listner to receive incoming events
        let appclone = Arc::clone(&app);
        thread::spawn(move || {
            Self::listen(appclone, recv)
        });

        Ok(app)
    }

    fn listen(app: Arc<Instruct>, recv: Receiver<Event>) {

    }
}