import os
import re
import time
import glob
import torch
import torchaudio
import gradio as gr
from datetime import datetime

from generator import load_csm_1b, Segment
from utils import parse_conversation, load_prompt_segment


# =========================
# ENV (giữ như app gốc)
# =========================
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = "cpu"   # Mac mini M4 → giữ CPU để ổn định


# =========================
# PATHS
# =========================
VOICE_DIR = "/Volumes/SSD256/dev-projects/csm/voice_samples"
OUTPUT_DIR = "/Volumes/SSD256/dev-projects/csm/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD SPEAKERS
# =========================
def load_speakers():
    speakers = {}
    wavs = glob.glob(os.path.join(VOICE_DIR, "*.wav"))

    for wav in wavs:
        name = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(VOICE_DIR, name + ".txt")
        if os.path.exists(txt):
            speakers[name] = {
                "audio": wav,
                "text": txt
            }
    return speakers


SPEAKERS = load_speakers()
SPEAKER_NAMES = list(SPEAKERS.keys())


# =========================
# CORE ENGINE
# =========================
def run_conversation(
    speaker_a_name,
    speaker_b_name,
    conversation_text,
    file_prefix,
    merge_audio,
    love_my_mac,
    pause_every,
    pause_seconds,
):
    print("Speaker A:", speaker_a_name)
    print("Speaker B:", speaker_b_name)

    generator = load_csm_1b(DEVICE)

    speaker_a = SPEAKERS[speaker_a_name]
    speaker_b = SPEAKERS[speaker_b_name]

    # ----- load prompt segments (GIỐNG APP GỐC) -----
    prompt_segments = [
        load_prompt_segment(speaker_a["text"], speaker_a["audio"], 0, generator.sample_rate),
        load_prompt_segment(speaker_b["text"], speaker_b["audio"], 1, generator.sample_rate),
    ]

    # ----- parse conversation -----
    conversation = parse_conversation(conversation_text)
    print("[CSM] Parsed", len(conversation), "turns")
    print(conversation)
    generated_segments = []
    saved_files = []

    for i, utterance in enumerate(conversation):
        print(f"Generating ({i+1}/{len(conversation)}): {utterance['text']}")

        try:
            with torch.inference_mode():
                audio_tensor = generator.generate(
                    text=utterance["text"],
                    speaker=utterance["speaker_id"],
                    context=prompt_segments + generated_segments,
                    max_audio_length_ms=10_000,
                    temperature = 1.0,
                    topk = 60
                    
                )

            if audio_tensor is None:
                print("⚠️ Returned None")
                continue

            audio_tensor = audio_tensor.cpu().float()

            seg = Segment(
                text=utterance["text"],
                speaker=utterance["speaker_id"],
                audio=audio_tensor
            )
            generated_segments.append(seg)

            ts = datetime.now().strftime("%d%H%M%S")
            fname = f"{file_prefix}_{i+1:02d}_{ts}.wav"
            out_path = os.path.join(OUTPUT_DIR, fname)

            torchaudio.save(out_path, audio_tensor.unsqueeze(0), generator.sample_rate)
            saved_files.append(out_path)

            # ----- Love my Mac mode -----
            if love_my_mac and (i + 1) % pause_every == 0:
                print(f"💤 Cooling down {pause_seconds}s ...")
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                time.sleep(pause_seconds)

        except Exception as e:
            print("❌ Error:", e)

    merged_path = None

    if merge_audio and len(generated_segments) > 0:
        all_audio = torch.cat([s.audio for s in generated_segments], dim=0)
        ts = datetime.now().strftime("%d%H%M%S")
        merged_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_FULL_{ts}.wav")
        torchaudio.save(merged_path, all_audio.unsqueeze(0), generator.sample_rate)

    return saved_files, merged_path


# =========================
# UI
# =========================
with gr.Blocks() as demo:
    gr.Markdown("# 🎙️ CSM Conversation Studio (Mac Edition)")

    with gr.Row():
        speaker_a_dd = gr.Dropdown(SPEAKER_NAMES, label="Speaker A (ID = 0)", value=SPEAKER_NAMES[0])
        speaker_b_dd = gr.Dropdown(SPEAKER_NAMES, label="Speaker B (ID = 1)", value=SPEAKER_NAMES[1])

    conversation_box = gr.Textbox(
        label="Conversation",
        lines=12,
        placeholder="[SPEAKER_00] Hello...\n[SPEAKER_01] Hi..."
    )

    file_prefix = gr.Textbox(label="File prefix", value="pod01")
    merge_audio = gr.Checkbox(label="Merge all segments after done", value=True)

    gr.Markdown("## 💻 Love my Mac")
    love_my_mac = gr.Checkbox(label="Enable Love my Mac mode", value=False)
    pause_every = gr.Slider(1, 10, value=3, step=1, label="Pause every X segments")
    pause_seconds = gr.Slider(1, 20, value=5, step=1, label="Pause Y seconds")

    run_btn = gr.Button("🚀 Generate Conversation")

    files_output = gr.File(label="Generated segments")
    merged_output = gr.File(label="Merged audio")

    run_btn.click(
        fn=run_conversation,
        inputs=[
            speaker_a_dd,
            speaker_b_dd,
            conversation_box,
            file_prefix,
            merge_audio,
            love_my_mac,
            pause_every,
            pause_seconds,
        ],
        outputs=[files_output, merged_output]
    )


if __name__ == "__main__":
    demo.launch(server_port=7866)
