# =========================
# run_csm_v1.py
# =========================
import os, time, glob
import torch
import torchaudio
import gradio as gr
from datetime import datetime

from generator import load_csm_1b, Segment
import config

# ---------- ENV FIXES FOR MAC ----------
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ---------- UTILS ----------
def scan_voice_samples():
    wavs = glob.glob(os.path.join(config.VOICE_SAMPLE_DIR, "*.wav"))
    pairs = {}
    for w in wavs:
        name = os.path.splitext(os.path.basename(w))[0]
        txt = os.path.join(config.VOICE_SAMPLE_DIR, name + ".txt")
        if os.path.exists(txt):
            pairs[name] = {"wav": w, "txt": txt}
    return pairs

VOICE_PAIRS = scan_voice_samples()


def load_prompt_audio(audio_path: str, target_sr: int):
    audio, sr = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio[0]
    else:
        audio = audio.squeeze(0)
    audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio


def prepare_prompt(pair_name: str, speaker_id: int, sample_rate: int):
    wav = VOICE_PAIRS[pair_name]["wav"]
    txt = VOICE_PAIRS[pair_name]["txt"]
    with open(txt, "r", encoding="utf-8") as f:
        text = f.read().strip()
    audio = load_prompt_audio(wav, sample_rate)
    return Segment(text=text, speaker=speaker_id, audio=audio)


# ---------- MODEL LOADING ----------

def load_generator():
    if config.RUNNING_MODE == "Offline":
        os.environ["HF_HOME"] = "/Volumes/SSD256/ai-models"
        print("[CSM] Offline mode: using local cache")

    print("[CSM] Loading model...")
    gen = load_csm_1b(config.DEVICE)
    return gen


generator = load_generator()


# ---------- CORE GENERATION ----------

def run_conversation(speaker_a, speaker_b, conversation_text, prefix, merge_audio,
                     love_my_mac, every_x, sleep_y):

    if not conversation_text.strip():
        return "⚠️ Conversation is empty", None

    prompt_a = prepare_prompt(speaker_a, 0, generator.sample_rate)
    prompt_b = prepare_prompt(speaker_b, 1, generator.sample_rate)

    lines = [l.strip() for l in conversation_text.split("\n") if l.strip()]
    conversation = []
    for i, line in enumerate(lines):
        speaker_id = 0 if i % 2 == 0 else 1
        conversation.append({"text": line, "speaker_id": speaker_id})

    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    outputs = []
    counter = 0

    for i, utt in enumerate(conversation):
        with torch.inference_mode():
            audio = generator.generate(
                text=utt["text"],
                speaker=utt["speaker_id"],
                context=prompt_segments + generated_segments,
                max_audio_length_ms=12_000,
            )

        if audio is None:
            continue

        audio = audio.cpu().float()
        seg = Segment(text=utt["text"], speaker=utt["speaker_id"], audio=audio)
        generated_segments.append(seg)

        ts = datetime.now().strftime("%d%H%M%S")
        filename = f"{prefix}_{i+1:02d}_{ts}.wav"
        outpath = os.path.join(config.OUTPUT_DIR, filename)

        torchaudio.save(outpath, audio.unsqueeze(0), generator.sample_rate)
        outputs.append(outpath)

        counter += 1
        if love_my_mac and counter % every_x == 0:
            torch.cuda.empty_cache()
            if hasattr(torch, "mps"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
            time.sleep(sleep_y)

    merged_path = None
    if merge_audio and len(generated_segments) > 1:
        all_audio = torch.cat([s.audio for s in generated_segments], dim=0)
        ts = datetime.now().strftime("%d%H%M%S")
        merged_name = f"{prefix}_FULL_{ts}.wav"
        merged_path = os.path.join(config.OUTPUT_DIR, merged_name)
        torchaudio.save(merged_path, all_audio.unsqueeze(0), generator.sample_rate)

    return "✅ Done", merged_path if merged_path else outputs


# ---------- GRADIO UI ----------

pair_names = list(VOICE_PAIRS.keys())

with gr.Blocks() as demo:
    gr.Markdown("# 🎙 CSM Conversation Generator – Mac Edition")

    with gr.Row():
        spk_a = gr.Dropdown(pair_names, label="Speaker A (wav + txt)")
        spk_b = gr.Dropdown(pair_names, label="Speaker B (wav + txt)")

    conversation_box = gr.Textbox(lines=10, label="Conversation (each line = one turn)")

    with gr.Row():
        prefix = gr.Textbox(value="pod01", label="Output prefix")
        merge_audio = gr.Checkbox(value=True, label="Merge all audio")

    with gr.Accordion("🍏 Love My Mac Mode", open=False):
        love_my_mac = gr.Checkbox(value=False, label="Enable Love My Mac")
        every_x = gr.Number(value=config.LOVE_MY_MAC_EVERY_X, precision=0, label="Every X segments")
        sleep_y = gr.Number(value=config.LOVE_MY_MAC_SLEEP_Y, precision=0, label="Sleep Y seconds")

    run_btn = gr.Button("🚀 Generate")
    status = gr.Textbox(label="Status")
    output_audio = gr.File(label="Generated audio")

    run_btn.click(
        run_conversation,
        inputs=[spk_a, spk_b, conversation_box, prefix, merge_audio, love_my_mac, every_x, sleep_y],
        outputs=[status, output_audio]
    )


demo.launch(server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT)
