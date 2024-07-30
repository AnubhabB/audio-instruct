// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use instruct::Instruct;
use tauri::Manager;

#[macro_use]
extern crate log;

mod commands;
mod instruct;
mod llama;
mod utils;
mod whisper;

fn main() {
    pretty_env_logger::init();

    let app = tauri::Builder::default()
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_dialog::init())
        .setup(|tauri_app| {
            let datadir = tauri_app.path().app_data_dir()?;
            // Initialize the instruct app
            let instruct = Instruct::new(datadir.clone()).expect("initialization failed");            
            tauri_app.manage(instruct);

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("Failed to build app!");
    
    // finally, lets run our app
    app.run(|_app_handle, event| {
        if let tauri::RunEvent::ExitRequested { api, .. } = event {
            warn!("exit requested {api:?}");
        }
    });
}
