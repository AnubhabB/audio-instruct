use std::{path::PathBuf, sync::{mpsc::{channel, Receiver, Sender}, Arc}, thread};

use anyhow::Result;

use crate::{commands::Response, llama::LlamaWrap, whisper::WhisperWrap};

/// A struct to maintain our app state
pub struct Instruct {
    /// Holds the app data directory
    datadir: PathBuf,
    /// holds the instantiated `Llama` quantized `gguf` model and associated methods
    llama: LlamaWrap,
    /// holds the `distil-whisper` model and the associated methods
    whisper: WhisperWrap,
    /// a channel for triggering Instruct methods through events
    send: Sender<Vec<f32>>,
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

    /// Exposes an API to send data into our MPSC channel
    pub fn send(&self, data: Vec<f32>) -> Result<()> {
        self.send.send(data)?;
        
        Ok(())
    }

    fn listen(app: Arc<Instruct>, recv: Receiver<Vec<f32>>) {
        while let Ok(next) = recv.recv() {
            
        }
    }

    pub fn text(&self, istruct: &str) -> Result<Response> {
        todo!()
    }

    pub fn audio(&self) -> Result<Response> {
        todo!()
    }
}