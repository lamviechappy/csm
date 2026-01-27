from pathlib import Path
from studio_engine import CSMStudioEngine
from generator import Segment
import torch
import torchaudio

def load_voice_prompt(file_path, speaker_id):
    """Load voice sample and convert to a clean CSM Segment."""
    audio, sr = torchaudio.load(file_path)

    # 1. Resample về 24k nếu cần
    if sr != 24000:
        audio = torchaudio.functional.resample(audio, sr, 24000)

    # 2. FORCE mono 1D
    from studio_engine import AudioIO
    audio = AudioIO._to_mono_1d(audio)

    # 3. Đưa về CPU float32
    audio = audio.cpu().float()

    print(f"[VOICE PROMPT] {file_path.name} shape:", audio.shape)

    return Segment(
        speaker=speaker_id,
        text="<VOICE_PROMPT>",
        audio=audio
    )


def main():
    project_name = "Test_Project_2026"
    engine = CSMStudioEngine(project_name=project_name)

    # ĐƯỜNG DẪN FILE MẪU (Theo yêu cầu của bạn)
    SAMPLE_DIR = Path("/Volumes/SSD256/dev-projects/csm/voice_samples")
    
    # Gán Voice Prompts làm "gốc" để giữ giọng ổn định
    print("\n[TEST] Loading Voice Prompts...")
    prompts = [
        load_voice_prompt(SAMPLE_DIR / "woman-1.wav", "speaker_0"),
        load_voice_prompt(SAMPLE_DIR / "man-1.wav", "speaker_1")
    ]
    
    # SỬA TẠI ĐÂY: Truy cập vào manager nằm trong GeneratorEngine
    if hasattr(engine, 'engine'):
        engine.engine.warmup_manager.voice_prompt_segments = prompts
    else:
        print("[ERROR] CSMStudioEngine không chứa engine thực thi!")    

    # Chạy Warmup
    print("\n🔥 Building global warmup (Lưu tại warmup_debug)...")
    warmup_segments = engine.prepare_warmup()

    # Chạy Generation với script test
    test_script = [
        {
            "speaker_id": "speaker_0",
            "text": "Hello, welcome back to My Podcast Studio"
        },
        {
            "speaker_id": "speaker_1",
            "text": "Hey, everyone. This is our second podcast. And I'm John."
        }
    ]
    
    print("\n🎙️ Generating Podcast...")
    engine.generate(test_script)

if __name__ == "__main__":
    main()
