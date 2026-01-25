import os
import torch
import torchaudio
from generator import Segment
from generator import load_csm_1b, Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "/Volumes/SSD256/dev-projects/csm/voice_samples copy/woman-1.wav",
    "/Volumes/SSD256/dev-projects/csm/voice_samples copy/man-1.wav",
    "/Volumes/SSD256/dev-projects/csm/voice_samples copy/woman-1.wav",
    "/Volumes/SSD256/dev-projects/csm/voice_samples copy/man-1.wav",
]
device = "cpu"
print(f"Using device: {device}")

# 3. LOAD MODEL
generator = load_csm_1b(device)
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)