<script lang="ts">
    import type { QuestionAnswer } from "$lib/types";

    export let qa: QuestionAnswer;
</script>

<div class="flex flex-col qa">
    <div class="flex flex-row caption">
        <div class="time">
            {qa.ts.toLocaleString()}
        </div>
        {#if qa.a == "__asking__"}
        <div style="color: red">Thinking ...</div>
        {/if}
        {#if qa.meta}
        <div class="flex flex-row" style="margin-left: auto; gap: 8px">
            <div>Tokens: {qa.meta.n_tokens}</div>
            <div>Time: {qa.meta.n_secs}s</div>
        </div>
        {/if}
    </div>
    <div class="question">
        {#if qa.mode == "text"}
        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M460-320v-280H340v-40h280v40H500v280h-40ZM95.38-95.38v-169.24H160v-430.76H95.38v-169.24h169.24V-800h430.76v-64.62h169.24v169.24H800v430.76h64.62v169.24H695.38V-160H264.62v64.62H95.38ZM264.62-200h430.76v-64.62H760v-430.76h-64.62V-760H264.62v64.62H200v430.76h64.62V-200ZM135.38-735.38h89.24v-89.24h-89.24v89.24Zm600 0h89.24v-89.24h-89.24v89.24Zm0 600h89.24v-89.24h-89.24v89.24Zm-600 0h89.24v-89.24h-89.24v89.24Zm89.24-600Zm510.76 0Zm0 510.76Zm-510.76 0Z"/></svg>
        {:else}
        <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960" width="24px" fill="#e8eaed"><path d="M480-440q-33.85 0-56.92-23.08Q400-486.15 400-520v-240q0-33.85 23.08-56.92Q446.15-840 480-840t56.92 23.08Q560-793.85 560-760v240q0 33.85-23.08 56.92Q513.85-440 480-440Zm0-200Zm-20 500v-140.69q-94-8.62-157-76.85-63-68.23-63-162.46h40q0 83 58.5 141.5T480-320q83 0 141.5-58.5T680-520h40q0 94.23-63 162.46t-157 76.85V-140h-40Zm20-340q17 0 28.5-11.5T520-520v-240q0-17-11.5-28.5T480-800q-17 0-28.5 11.5T440-760v240q0 17 11.5 28.5T480-480Z"/></svg>
        {/if}
        <span>{qa.q}</span>
    </div>
    {#if qa.a != "__asking__"}
    <div class="answer">
        {qa.a}
    </div>
    {/if}
</div>

<style>
.qa {
    gap: 2;
    padding: 24px 0;
    border-bottom: 1px solid rgba(166,166,166, .75);
}

.qa .caption {
    padding: 4px 0;
}

.qa .caption * {
    font-size: 0.6rem;
}

.qa .question, .qa .answer {
    font-size: 1rem;
}

.qa .question {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 12px;
    font-weight: bold;
    color: #5b8dee;
}

.qa .answer {
    padding: 8px 0;
}
</style>