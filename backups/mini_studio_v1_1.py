import os
import json
import torch
import torchaudio
from datetime import datetime
from dataclasses import dataclass

# Tokenizers chạy đơn luồng để tránh xung đột.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# =========================
# ⚠️ IMPORT CSM CORE
# =========================
# Bạn phải có sẵn các hàm này từ CSM gốc
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

    # 1. Thử JSON trước
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [{"text": x["text"], "speaker_id": int(x["speaker_id"])} for x in data]
    except Exception:
        pass  # không phải JSON → fallback

    # 2. Parse dạng [SPEAKER_xx]
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
    print("🎙 MINI STUDIO V1 – VOICE LAB")
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

    # 4. LOAD MODEL
    device = "cpu"
    print("\n🚀 Loading CSM model...")
    generator = load_csm_1b(device)
    print("✅ Model loaded")

    # 5. PREPARE VOICE PROMPTS (ANCHOR)
    print("\n🎤 Loading voice prompts...")
    prompt_0 = prepare_prompt("Voice anchor speaker 0", 0, wav0, generator.sample_rate)
    prompt_1 = prepare_prompt("Voice anchor speaker 1", 1, wav1, generator.sample_rate)
    voice_prompts = [prompt_0, prompt_1]
    print("✅ DONE")

    # 6. GENERATION LOOP
    generated_segments = []

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
            generated_segments.append(
                Segment(text=utt["text"], speaker=utt["speaker_id"], audio=audio)
            )

        except Exception as e:
            print("❌ Error:", e)

    # 7. CONCAT & SAVE
    if not generated_segments:
        print("❌ Không có audio nào được tạo.")
        return

    print("\n🔗 Concatenating audio...")
    full_audio = torch.cat([s.audio for s in generated_segments], dim=0)

    torchaudio.save(out_path, full_audio.unsqueeze(0), generator.sample_rate)
    print(f"\n✅ DONE. Full file: {out_path}\n")


if __name__ == "__main__":
    main()
