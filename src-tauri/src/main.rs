// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[macro_use]
extern crate log;

mod commands;
mod instruct;
mod llama;
mod utils;
mod whisper;

fn main() {
    pretty_env_logger::init();

    todo!()
}
