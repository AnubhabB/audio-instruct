export interface Meta {
    n_tokens: number,
    n_secs: number
}

export interface QuestionAnswer {
    q: string,
    a: string,
    ts: Date,
    meta?: Meta,
    mode: "audio"|"text"
}

export interface Inference {
    instruct: string,
    text: string,
    meta?: Meta
}