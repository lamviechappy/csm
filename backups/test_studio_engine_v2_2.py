from pathlib import Path
from studio_engine import CSMStudioEngine
from generator import Segment
import torch
import torchaudio


# ----------------------------
# Load voice prompt helper
# ----------------------------
def load_voice_prompt(file_path: Path, speaker_id: str):
    """Load voice sample wav -> Segment (mono 1D)"""
    audio, sr = torchaudio.load(file_path)

    # Nếu stereo -> mono
    if audio.ndim == 2:
        audio = audio.mean(dim=0)

    # Ép float32 CPU
    audio = audio.cpu().float()

    print(f"[VOICE PROMPT] {file_path.name} shape: {audio.shape}")

    return Segment(
        speaker=speaker_id,
        text="<VOICE_PROMPT>",
        audio=audio
    )


# ----------------------------
# MAIN
# ----------------------------
def main():
    project_name = "Test_Project_2026"
    engine = CSMStudioEngine(project_name=project_name)

    SAMPLE_DIR = Path("/Volumes/SSD256/dev-projects/csm/voice_samples")

    # ============================
    # 1. LOAD VOICE PROMPTS
    # ============================
    print("\n[TEST] Loading Voice Prompts...")
    prompts = [
        load_voice_prompt(SAMPLE_DIR / "woman-1.wav", "speaker_0"),
        load_voice_prompt(SAMPLE_DIR / "man-1.wav", "speaker_1")
    ]

    # Gán trực tiếp cho warmup_manager
    engine.engine.warmup_manager.voice_prompt_segments = prompts

    # ============================
    # 2. BUILD GLOBAL WARMUP
    # ============================
    print("\n🔥 Building global warmup (debug files will be saved)...")
    warmup_segments = engine.prepare_warmup()

    if not warmup_segments:
        print("❌ Warmup failed or empty. STOP.")
        return

    print(f"✅ Warmup ready: {len(warmup_segments)} segments")

    # ============================
    # 3. GENERATE PODCAST
    # ============================
    test_script = [
        {
            "speaker_id": "speaker_0",
            "text": "Hello, welcome back to My Podcast Studio."
        },
        {
            "speaker_id": "speaker_1",
            "text": "Hey everyone. This is our second podcast, and I'm John."
        }
    ]

    print("\n🎙️ Generating Podcast...")
    engine.generate(test_script)

    print("\n==============================")
    print("✅ TEST COMPLETED SUCCESSFULLY")
    print("==============================")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Running on device: {device}")
    main()
