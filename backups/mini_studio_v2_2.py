import os
import json
import time
import gc
import torch
import torchaudio
from datetime import datetime
from dataclasses import dataclass

from generator import load_csm_1b
from generator import Segment

# Tokenizers chạy đơn luồng để tránh xung đột.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

VOICE_SAMPLE_DIR = "voice_samples"
DEFAULT_OUTPUT_DIR = "outputs"


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

    # Try JSON first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [{"text": x["text"], "speaker_id": int(x["speaker_id"])} for x in data]
    except Exception:
        pass

    # Fallback: [SPEAKER_xx] format
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
    print("🎙 MINI STUDIO V2.2 – LOVE MY MAC EDITION")
    print("==============================\n")

    # 1. INPUT CONVERSATION
    conversation_raw = input_multiline(
        "👉 Paste conversation (JSON or [SPEAKER_xx]). Empty line to finish:"
    )

    conversation = parse_conversation(conversation_raw)
    if not conversation:
        print("❌ Cannot parse conversation.")
        return

    # 2. SPEAKER PROMPTS
    spk0 = input("👉 SPEAKER_00 voice sample name (no .wav): ").strip()
    spk1 = input("👉 SPEAKER_01 voice sample name (no .wav): ").strip()

    wav0 = os.path.join(VOICE_SAMPLE_DIR, f"{spk0}.wav")
    wav1 = os.path.join(VOICE_SAMPLE_DIR, f"{spk1}.wav")

    if not os.path.exists(wav0) or not os.path.exists(wav1):
        print("❌ Voice sample not found in voice_samples/")
        return

    # 3. OUTPUT CONFIG
    out_dir = input(f"👉 Output folder (Enter = {DEFAULT_OUTPUT_DIR}): ").strip()
    if not out_dir:
        out_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    out_name = input("👉 Final file name (Enter = mini_ddhhmmss.wav): ").strip()
    if not out_name:
        out_name = "mini_" + datetime.now().strftime("%d%H%M%S") + ".wav"
    if not out_name.endswith(".wav"):
        out_name += ".wav"

    final_path = os.path.join(out_dir, out_name)
    block_dir = os.path.join(out_dir, "_blocks")
    os.makedirs(block_dir, exist_ok=True)

    # 4. LOVE-MY-MAC MODE
    love_mode = input("👉 Love-my-mac mode? (y/n, Enter = n): ").strip().lower() == "y"

    cool_every = None
    cool_seconds = None

    if love_mode:
        cool_every = input("👉 Cooldown every how many turns? (Enter = 10): ").strip()
        cool_seconds = input("👉 Cooldown how many seconds? (Enter = 5): ").strip()

        cool_every = int(cool_every) if cool_every else 10
        cool_seconds = int(cool_seconds) if cool_seconds else 5

        print(f"❤️ Love-my-mac ON | every {cool_every} turns → rest {cool_seconds}s")

    # 5. LOAD MODEL
    print("\n🚀 Loading CSM model...")
    device = "cpu"
    generator = load_csm_1b(device)
    print("✅ Model loaded")

    # 6. LOAD VOICE PROMPTS
    print("\n🎤 Loading voice prompts...")
    prompt_0 = prepare_prompt("Voice anchor speaker 0", 0, wav0, generator.sample_rate)
    prompt_1 = prepare_prompt("Voice anchor speaker 1", 1, wav1, generator.sample_rate)
    voice_prompts = [prompt_0, prompt_1]
    print("✅ DONE")

    # 7. GENERATION LOOP (SAFE)
    generated_segments = []
    block_segments = []
    block_index = 1

    try:
        for i, utt in enumerate(conversation):
            print(f"\n🧩 Generating {i+1}/{len(conversation)} | Speaker {utt['speaker_id']}")

            sliding_context = generated_segments[-2:]
            context = voice_prompts + sliding_context

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
            block_segments.append(seg)

            # Save block every 10 turns
            if len(block_segments) == 10:
                block_path = os.path.join(block_dir, f"block_{block_index:03d}.wav")
                block_audio = torch.cat([s.audio for s in block_segments], dim=0)
                torchaudio.save(block_path, block_audio.unsqueeze(0), generator.sample_rate)
                print(f"💾 Saved block: {block_path}")
                block_segments.clear()
                block_index += 1

            # LOVE MY MAC
            if love_mode and (i + 1) % cool_every == 0:
                print("❤️ Cooling down...")
                time.sleep(cool_seconds)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user. Entering safe exit...")

    # 8. SAVE LAST BLOCK
    if block_segments:
        block_path = os.path.join(block_dir, f"block_{block_index:03d}.wav")
        block_audio = torch.cat([s.audio for s in block_segments], dim=0)
        torchaudio.save(block_path, block_audio.unsqueeze(0), generator.sample_rate)
        print(f"💾 Saved last block: {block_path}")

    # 9. MERGE ALL
    if not generated_segments:
        print("❌ No audio generated.")
        return

    print("\n🔗 Merging all segments...")
    full_audio = torch.cat([s.audio for s in generated_segments], dim=0)
    torchaudio.save(final_path, full_audio.unsqueeze(0), generator.sample_rate)

    print(f"\n✅ DONE. Final file: {final_path}")
    print("❤️ Love your Mac. It loves you back.\n")


if __name__ == "__main__":
    main()
