// Adapted from https://github.com/kgullion/vite-typescript-audio-worklet-example/blob/main/src/main.ts

declare class AudioWorkletProcessor {
    constructor(options?: AudioWorkletNodeOptions);
    readonly port: MessagePort;
    process(
        inputs: Float32Array[][],
        outputs: Float32Array[][],
        parameters: Record<string, Float32Array>
    ): boolean;
}

declare function registerProcessor(
    name: string,
    processorCtor: (new (options?: AudioWorkletNodeOptions) => AudioWorkletProcessor)
): void;