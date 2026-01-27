import os
import re
import time
import torch
import torchaudio
import gradio as gr
from datetime import datetime
from dataclasses import dataclass

# =========================
# ⚠️ IMPORT CSM CORE
# =========================
from generator import load_csm_1b
from generator import Segment

# =========================
# 🎯 CONFIG
# =========================
VOICE_SAMPLE_DIR = "voice_samples"

# =========================
# 📦 DATA
# =========================
@dataclass
class VoicePrompt:
    name: str
    wav_path: str
    txt_path: str

# =========================
# 🎤 AUDIO UTILS
# =========================
def load_audio(wav_path, target_sr):
    wav, sr = torchaudio.load(wav_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0).contiguous().float()


def prepare_prompt_from_pair(speaker_id, wav_path, txt_path, target_sr):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    audio = load_audio(wav_path, target_sr)
    print(f"[VOICE PROMPT] {os.path.basename(wav_path)} | {audio.shape}")

    return Segment(text=text, speaker=speaker_id, audio=audio)


# =========================
# 🧠 CONVERSATION PARSER
# Only accept:
# [SPEAKER_00] ...
# [SPEAKER_01] ...
# =========================
def parse_conversation(raw: str):
    conversation = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        m = re.match(r"\[SPEAKER_(\d+)\](.*)", line)
        if not m:
            continue

        speaker_id = int(m.group(1))
        text = m.group(2).strip()

        conversation.append({
            "speaker_id": speaker_id,
            "text": text
        })

    return conversation


# =========================
# 🎧 VOICE SAMPLE SCAN
# Only return voices that have BOTH wav + txt
# =========================
def scan_voice_samples():
    if not os.path.exists(VOICE_SAMPLE_DIR):
        return []

    files = os.listdir(VOICE_SAMPLE_DIR)
    wavs = {os.path.splitext(f)[0] for f in files if f.endswith(".wav")}
    txts = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}

    valid = sorted(list(wavs & txts))
    return valid


# =========================
# 🔢 ORDER NUMBER TOOLS
# =========================
def insert_order_no(text):
    lines = []
    idx = 1
    for line in text.splitlines():
        if line.strip():
            if not re.match(r"\[\d+\]", line):
                line = f"[{idx:03d}] {line}"
                idx += 1
        lines.append(line)
    return "\n".join(lines)


def delete_order_no(text):
    return "\n".join([re.sub(r"^\[\d+\]\s*", "", l) for l in text.splitlines()])


# =========================
# 🧠 GLOBAL MODEL
# =========================
GENERATOR = None

def load_model():
    global GENERATOR
    if GENERATOR is None:
        print("🚀 Loading CSM model...")
        GENERATOR = load_csm_1b("cpu")
        print("✅ Model loaded")
    return GENERATOR


# =========================
# 🚀 GENERATION CORE
# =========================
def run_generation(conversation_raw,
                   spk0_name,
                   spk1_name,
                   block_mode,
                   output_dir):

    yield "🚀 Loading model...", None
    generator = load_model()

    # -------- folders --------
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(output_dir, f"session_{ts}")
    seg_dir = os.path.join(base_dir, "segments")
    block_dir = os.path.join(base_dir, "blocks")
    final_dir = os.path.join(base_dir, "final")

    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(block_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)

    # -------- parse --------
    conversation = parse_conversation(conversation_raw)
    if not conversation:
        yield "❌ Conversation parse failed.", None
        return

    # -------- load voice anchors (STRICT) --------
    wav0 = os.path.join(VOICE_SAMPLE_DIR, spk0_name + ".wav")
    txt0 = os.path.join(VOICE_SAMPLE_DIR, spk0_name + ".txt")
    wav1 = os.path.join(VOICE_SAMPLE_DIR, spk1_name + ".wav")
    txt1 = os.path.join(VOICE_SAMPLE_DIR, spk1_name + ".txt")

    if not all(os.path.exists(p) for p in [wav0, txt0, wav1, txt1]):
        yield "❌ Voice sample must include BOTH wav + txt.", None
        return

    prompt_0 = prepare_prompt_from_pair(0, wav0, txt0, generator.sample_rate)
    prompt_1 = prepare_prompt_from_pair(1, wav1, txt1, generator.sample_rate)
    voice_prompts = [prompt_0, prompt_1]

    # -------- block config --------
    block_size = 5 if block_mode == "5 turns" else 10

    generated_segments = []
    block_segments = []
    all_blocks = []
    latest_block_path = None

    # -------- generation loop --------
    for i, utt in enumerate(conversation):
        yield f"🧩 Generating {i+1}/{len(conversation)} | Speaker {utt['speaker_id']}", latest_block_path

        sliding_context = generated_segments[-2:]
        context = voice_prompts + sliding_context

        with torch.inference_mode():
            audio = generator.generate(
                text=utt["text"],
                speaker=utt["speaker_id"],
                context=context,
                max_audio_length_ms=12_000
            )

        audio = audio.cpu().float()
        seg = Segment(text=utt["text"], speaker=utt["speaker_id"], audio=audio)

        generated_segments.append(seg)
        block_segments.append(seg)

        # save segment
        seg_path = os.path.join(seg_dir, f"seg_{i+1:03d}_spk{utt['speaker_id']}.wav")
        torchaudio.save(seg_path, audio.unsqueeze(0), generator.sample_rate)

        # block export
        if len(block_segments) == block_size:
            block_audio = torch.cat([s.audio for s in block_segments], dim=0)
            block_id = len(all_blocks) + 1
            latest_block_path = os.path.join(block_dir, f"block_{block_id:03d}.wav")
            torchaudio.save(latest_block_path, block_audio.unsqueeze(0), generator.sample_rate)
            all_blocks.append(latest_block_path)
            block_segments = []
            yield f"✅ Block {block_id:03d} exported.", latest_block_path

    # last block
    if block_segments:
        block_audio = torch.cat([s.audio for s in block_segments], dim=0)
        block_id = len(all_blocks) + 1
        latest_block_path = os.path.join(block_dir, f"block_{block_id:03d}.wav")
        torchaudio.save(latest_block_path, block_audio.unsqueeze(0), generator.sample_rate)
        all_blocks.append(latest_block_path)

    # final concat
    final_audio = torch.cat(
        [torchaudio.load(b)[0].squeeze(0) for b in all_blocks], dim=0
    )
    final_path = os.path.join(final_dir, "final.wav")
    torchaudio.save(final_path, final_audio.unsqueeze(0), generator.sample_rate)

    yield f"🎉 DONE. Final saved: {final_path}", latest_block_path


# =========================
# 🖥️ GUI
# =========================
voices = scan_voice_samples()

with gr.Blocks(title="Mini Studio GUI v1") as demo:
    gr.Markdown("# 🎙 Mini Studio – Voice Lab (GUI v1)")
    gr.Markdown("CSM conversation studio – strict voice anchoring (wav + txt)")

    with gr.Row():
        with gr.Column(scale=6):
            convo_input = gr.Textbox(lines=22, label="Conversation Input")
            with gr.Row():
                btn_insert = gr.Button("➕ Insert Order No.")
                btn_delete = gr.Button("➖ Delete Order No.")

        with gr.Column(scale=4):
            spk0_dd = gr.Dropdown(voices, label="Speaker 0 voice")
            spk1_dd = gr.Dropdown(voices, label="Speaker 1 voice")
            block_mode = gr.Radio(["5 turns", "10 turns"], value="5 turns", label="Block size")
            output_dir = gr.Textbox(label="Output directory", value=os.path.abspath("outputs"))
            run_btn = gr.Button("🚀 Generate", variant="primary")
            status = gr.Markdown("Idle.")
            latest_audio = gr.Audio(label="Latest block audio", autoplay=True)

    btn_insert.click(insert_order_no, convo_input, convo_input)
    btn_delete.click(delete_order_no, convo_input, convo_input)

    run_btn.click(
        run_generation,
        [convo_input, spk0_dd, spk1_dd, block_mode, output_dir],
        [status, latest_audio]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_port=7866, share=False, allowed_paths=["/Volumes/SSD256"])
