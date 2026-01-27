import os
import json
import torch
import torchaudio
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
DEFAULT_OUTPUT_DIR = "outputs"


# =========================
# 📦 DATA
# =========================
@dataclass
class VoicePrompt:
    name: str
    wav_path: str
    text_path: str | None


# =========================
# 🎤 AUDIO UTILS
# =========================
def load_audio(wav_path, target_sr):
    wav, sr = torchaudio.load(wav_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    wav = wav.squeeze(0).contiguous().float()
    return wav


def prepare_prompt(text, speaker_id, wav_path, target_sr):
    audio = load_audio(wav_path, target_sr)
    print(f"[VOICE PROMPT] {os.path.basename(wav_path)} shape: {audio.shape}")
    return Segment(text=text, speaker=speaker_id, audio=audio)


# =========================
# 🧠 CONVERSATION PARSER
# =========================
def parse_conversation(raw: str):
    raw = raw.strip()

    # 1. Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [{"text": x["text"], "speaker_id": int(x["speaker_id"])} for x in data]
    except Exception:
        pass

    # 2. Parse [SPEAKER_xx]
    conversation = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("[SPEAKER_") and "]" in line:
            tag, text = line.split("]", 1)
            speaker_id = int(tag.replace("[SPEAKER_", "").replace("]", ""))
            conversation.append({
                "text": text.strip(),
                "speaker_id": speaker_id
            })

    return conversation


# =========================
# 🖥️ INPUT
# =========================
def input_multiline(prompt):
    print(prompt)
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)
    return "\n".join(lines)


# =========================
# 🚀 MAIN
# =========================
def main():
    print("\n==============================")
    print("🎙 MINI STUDIO V2 – BLOCK MODE")
    print("==============================\n")

    # 1. INPUT CONVERSATION
    conversation_raw = input_multiline(
        "👉 Dán conversation (JSON hoặc [SPEAKER_xx]...). Kết thúc bằng dòng trống:"
    )

    conversation = parse_conversation(conversation_raw)
    if not conversation:
        print("❌ Không đọc được conversation.")
        return

    # 2. INPUT SPEAKER NAMES
    spk0 = input("👉 Tên voice mẫu SPEAKER_00 (không .wav): ").strip()
    spk1 = input("👉 Tên voice mẫu SPEAKER_01 (không .wav): ").strip()

    wav0 = os.path.join(VOICE_SAMPLE_DIR, f"{spk0}.wav")
    wav1 = os.path.join(VOICE_SAMPLE_DIR, f"{spk1}.wav")

    if not os.path.exists(wav0) or not os.path.exists(wav1):
        print("❌ Không tìm thấy file wav trong voice_samples/")
        return

    # 3. OUTPUT CONFIG
    out_dir = input(f"👉 Thư mục output (Enter = {DEFAULT_OUTPUT_DIR}): ").strip()
    if not out_dir:
        out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_name = input("👉 Tên file output (Enter = mini_ddhhmmss.wav): ").strip()
    if not out_name:
        out_name = "mini_" + datetime.now().strftime("%d%H%M%S") + ".wav"
    if not out_name.endswith(".wav"):
        out_name += ".wav"

    out_path = os.path.join(out_dir, out_name)

    # 4. BLOCK SIZE
    x_raw = input("👉 Số turns mỗi block (Enter = 10): ").strip()
    if not x_raw:
        block_size = 10
    else:
        block_size = int(x_raw)

    print(f"🧱 Block size = {block_size} turns")

    # 5. LOAD MODEL
    device = "cpu"
    print("\n🚀 Loading CSM model...")
    generator = load_csm_1b(device)
    print("✅ Model loaded")

    # 6. PREPARE VOICE PROMPTS
    print("\n🎤 Loading voice prompts...")
    prompt_0 = prepare_prompt("Voice anchor speaker 0", 0, wav0, generator.sample_rate)
    prompt_1 = prepare_prompt("Voice anchor speaker 1", 1, wav1, generator.sample_rate)
    voice_prompts = [prompt_0, prompt_1]
    print("✅ DONE")

    # 7. GENERATION LOOP
    generated_segments = []

    block_segments = []
    block_paths = []
    block_index = 1

    for i, utt in enumerate(conversation):
        print(f"\n🧩 Generating {i+1}/{len(conversation)} | Speaker {utt['speaker_id']}")

        sliding_context = generated_segments[-2:]
        context = voice_prompts + sliding_context

        try:
            with torch.inference_mode():
                audio = generator.generate(
                    text=utt["text"],
                    speaker=utt["speaker_id"],
                    context=context,
                    max_audio_length_ms=12_000
                )

            if audio is None:
                print("⚠️ Returned None")
                continue

            audio = audio.cpu().float()

            seg = Segment(text=utt["text"], speaker=utt["speaker_id"], audio=audio)
            generated_segments.append(seg)
            block_segments.append(audio)

            # SAVE BLOCK
            if len(block_segments) == block_size:
                block_audio = torch.cat(block_segments, dim=0)
                block_path = os.path.join(out_dir, f"block_{block_index:03d}.wav")
                torchaudio.save(block_path, block_audio.unsqueeze(0), generator.sample_rate)

                print(f"💾 Saved block: {block_path}")

                block_paths.append(block_path)
                block_segments = []
                block_index += 1

        except Exception as e:
            print("❌ Error:", e)

    # SAVE LAST BLOCK
    if block_segments:
        block_audio = torch.cat(block_segments, dim=0)
        block_path = os.path.join(out_dir, f"block_{block_index:03d}.wav")
        torchaudio.save(block_path, block_audio.unsqueeze(0), generator.sample_rate)
        print(f"💾 Saved block: {block_path}")
        block_paths.append(block_path)

    if not block_paths:
        print("❌ Không có audio block nào được tạo.")
        return

    # 8. FINAL CONCAT
    print("\n🔗 Concatenating final audio...")
    all_audio = []

    for p in block_paths:
        wav, _ = torchaudio.load(p)
        all_audio.append(wav.squeeze(0))

    final_audio = torch.cat(all_audio, dim=0)
    torchaudio.save(out_path, final_audio.unsqueeze(0), generator.sample_rate)

    print(f"\n✅ DONE. Full file: {out_path}\n")


if __name__ == "__main__":
    main()
