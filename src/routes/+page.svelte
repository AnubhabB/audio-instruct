<script lang="ts">
  // Adapted from https://github.com/kgullion/vite-typescript-audio-worklet-example/blob/main/src/main.ts
  import audioProcUrl from "$lib/audio-proc/audio-processor?url";
  import { invoke } from '@tauri-apps/api/core';
  import type { Inference, QuestionAnswer } from "$lib/types";
  import Qa from "./QA.svelte";

const BUFFER_SIZE = 4096;
const SAMPLE_RATE = 16000; // Whisper typically expects 16kHz audio

let qas: QuestionAnswer[] = [];
let question: string,
  asking: boolean = false,
  isrecording: boolean = false,
  recordstart: Date|null = null;

let stream: MediaStream|null = null;

let audioContext: AudioContext|null = null;
let source: MediaStreamAudioSourceNode|null = null;
let workletNode: AudioWorkletNode|null = null;

const record = async () => {
  if(stream) {
      console.error("Duplicate record??");
      return;
  }

  stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // Create AudioContext
  audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });

  // Load and register the audio worklet
  await audioContext.audioWorklet.addModule(audioProcUrl)

  // Create MediaStreamSource
  source = audioContext.createMediaStreamSource(stream);

  // Create AudioWorkletNode
  workletNode = new AudioWorkletNode(audioContext, 'audio-processor', {
      outputChannelCount: [1],
      processorOptions: {
          bufferSize: BUFFER_SIZE
      }
  });

  // Connect the nodes
  source.connect(workletNode);
  workletNode.connect(audioContext.destination);

  // Set up message handling from the audio worklet
  workletNode.port.onmessage = handleAudioData;
}

const handleAudioData = async (event: MessageEvent): Promise<void> => {
  // Step 2 & 3: Receive Float32Array data and convert to PCM
  const float32Array = event.data as Float32Array;

  invoke("audio_chunk", float32Array);
}

const stopRecord = async () => {
  goAskAudio();

  if(workletNode) {
      workletNode.disconnect();
      workletNode = null;
  }

  if(source) {
      source.disconnect();
      source = null;
  }

  if(audioContext) {
      audioContext.close();
      audioContext = null;
  }

  if(stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
  }

  recordstart = null;
}

const toggleRecord = async () => {
  isrecording = !isrecording;
  if(isrecording) {
      recordstart = new Date();
      record();
  } else {
      stopRecord()
  }
}

const command = async (text?: string, audio?: boolean) => {
  let cmd = { text: text, audio: audio };
  let res: Inference = await invoke("ask", { cmd });
  
  let idx = qas.length - 1;
  let qa: QuestionAnswer = qas[idx];
  
  qa.q = res.instruct;
  qa.a = res.text;
  qa.meta = res.meta;

  qas = [...qas];

  asking = false;
}

const goAskText = async () => {
  asking = true;
  // We are just using a simple keyword to 
  qas.push({ q: question, a: "__asking__", ts: new Date(), mode: "text" });
  question = "";

  qas = [...qas];

  // The inference generation is extremely resource intensive, giving our UI to update before the call
  setTimeout(() => {
      command(qas[qas.length - 1].q, false)
  }, 100)
}

const goAskAudio = async () => {
  asking = true;
  // We are just using a simple keyword to 
  qas.push({ q: "..", a: "__asking__", ts: new Date(), mode: "audio" });
  question = "";

  qas = [...qas];

  // The inference generation is extremely resource intensive, giving our UI to update before the call
  setTimeout(() => {
      command(undefined, true)
  }, 100)
}

</script>

<div class="canvas flex flex-col">
  <div class="grid" style="grid-template-columns: 70% 30%; gap: 24px">
      <div class="input flex center relative">
          <input
              type="text"
              bind:value={question}
              on:keyup={(e) => { if(e.key == "Enter") goAskText() }}
              class="input full" placeholder="Ask your question!"
              disabled={asking}
          />
      </div>
      <div class="flex flex-row center">
          <button style="width: 96px; height: 96px; background-color: rgb(88, 117, 247); border: none; outline: none; border-radius: 50%; cursor: pointer" class="flex center justify" disabled={asking} on:click={toggleRecord}>
              {#if !isrecording}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width=64 height=64><title>microphone</title><path fill="white" d="M12,2A3,3 0 0,1 15,5V11A3,3 0 0,1 12,14A3,3 0 0,1 9,11V5A3,3 0 0,1 12,2M19,11C19,14.53 16.39,17.44 13,17.93V21H11V17.93C7.61,17.44 5,14.53 5,11H7A5,5 0 0,0 12,16A5,5 0 0,0 17,11H19Z" /></svg>
              {:else}
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width=64 height=64><title>stop-circle-outline</title><path fill="white" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4C16.41,4 20,7.59 20,12C20,16.41 16.41,20 12,20C7.59,20 4,16.41 4,12C4,7.59 7.59,4 12,4M9,9V15H15V9" /></svg>
              {/if}
          </button>
      </div>
  </div>
  {#each [...qas].reverse() as qa}
      <Qa qa={qa}/>
  {/each}
</div>


<style>
.canvas {
  width: 90%;
  height: 100vh;
  padding: 24px;
  max-width: 2048px;
}

.input {
  width: 100%;
}
</style>