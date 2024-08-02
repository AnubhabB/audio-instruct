use std::{
    path::PathBuf,
    sync::{
        mpsc::{channel, Receiver, Sender},
        Arc,
    },
    thread,
};

use anyhow::Result;

use crate::{commands::Response, llama::LlamaWrap, whisper::WhisperWrap};

/// A struct to maintain our app state
pub struct Instruct {
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
            llama,
            whisper,
            send: send.clone(),
        });

        // spawn a listner to receive incoming events
        let appclone = Arc::clone(&app);
        thread::spawn(move || Self::listen(appclone, recv));

        Ok(app)
    }

    /// Exposes an API to send data into our MPSC channel
    pub fn send(&self, data: Vec<f32>) -> Result<()> {
        self.send.send(data)?;

        Ok(())
    }

    fn listen(app: Arc<Instruct>, recv: Receiver<Vec<f32>>) {
        while let Ok(next) = recv.recv() {
            app.whisper.chunk(next);
        }
    }

    /// Public API to call text inference
    pub fn text(&self, instruct: &str) -> Result<Response> {
        let (txt, n_tokens, elapsed) = self.llama.infer(instruct)?;

        Ok(Response::new(
            instruct,
            &txt,
            n_tokens as u32,
            elapsed.as_secs(),
        ))
    }

    /// Public API to trigger audio inference
    pub fn audio(&self) -> Result<Response> {
        let (transcript, n_tokens, elapsed) = self.whisper.infer()?;
        let (generated, n_txt_tok, txt_elapsed) = self.llama.infer(&transcript)?;

        Ok(Response::new(
            &transcript,
            &generated,
            (n_tokens + n_txt_tok) as u32,
            (elapsed + txt_elapsed).as_secs(),
        ))
    }
}
