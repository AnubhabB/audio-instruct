use std::{sync::{mpsc::{channel, Receiver, Sender}, Arc}, thread};

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
    /// holds the instantiated `Llama` quantized `gguf` model and associated methods
    llama: LlamaWrap,
    /// holds the `distil-whisper` model and the associated methods
    whisper: WhisperWrap,
    /// a channel for triggering Instruct methods through events
    send: Sender<Event>,
}

impl Instruct {
    pub fn init() -> Result<(Arc<Self>, Sender<Event>)> {
        let llama = LlamaWrap::new()?;
        let whisper = WhisperWrap::new()?;

        let (send, recv) = channel();

        let app = Arc::new(Self {
            llama, whisper, send: send.clone()
        });

        // spawn a listner to receive incoming events
        let appclone = Arc::clone(&app);
        thread::spawn(move || {
            Self::listen(appclone, recv)
        });

        todo!()
    }

    fn listen(app: Arc<Instruct>, recv: Receiver<Event>) {

    }
}