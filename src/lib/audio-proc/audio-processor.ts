// Adapted from https://github.com/kgullion/vite-typescript-audio-worklet-example/blob/main/src/main.ts

class AudioProcessor extends AudioWorkletProcessor {
    private bufferSize: number;
    private buffer: Float32Array;
    private bufferIndex: number;

    constructor(options?: AudioWorkletNodeOptions) {
        super();
        this.bufferSize = options?.processorOptions.bufferSize || 4096;
        this.buffer = new Float32Array(this.bufferSize);
        this.bufferIndex = 0;
    }

    process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean {
        const input = inputs[0];
        const channel = input[0];

        if (channel) {
            for (let i = 0; i < channel.length; i++) {
                this.buffer[this.bufferIndex++] = channel[i];

                if (this.bufferIndex === this.bufferSize) {
                    this.port.postMessage(this.buffer);
                    this.buffer = new Float32Array(this.bufferSize);
                    this.bufferIndex = 0;
                }
            }
        }

        return true;
    }
}

registerProcessor('audio-processor', AudioProcessor);