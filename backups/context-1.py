import os
import torch
import torchaudio
# Giả định bạn đã cài đặt csm qua pip hoặc clone từ github
from csm import CSM 

# 1. Khởi tạo model (generator)
# Đảm bảo bạn đã tải checkpoint và cấu hình đúng thiết bị (cuda hoặc cpu)
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = CSM.from_pretrained("sesame/csm-1b").to(device)

speakers = [0, 1, 0, 0]
transcripts = [
    "hey how are you doing.",
    "pretty good, pretty good.",
    "i'm great.",
    "so happy to be speaking to you.",
]
# Lưu ý: Sửa lại đường dẫn file nếu cần thiết
audio_paths = [
    "/volumes/ssd256/dev-projects/csm/voice_samples copy/woman-1.wav",
    "/volumes/ssd256/dev-projects/csm/voice_samples copy/man-1.wav",
    "/volumes/ssd256/dev-projects/csm/voice_samples copy/woman-1.wav",
    "/volumes/ssd256/dev-projects/csm/voice_samples copy/man-1.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    # Chuyển về mono nếu là stereo
    if audio_tensor.shape[0] > 1:
        audio_tensor = torch.mean(audio_tensor, dim=0, keepdim=True)
    
    # Resample về sample_rate của generator (thường là 24kHz hoặc 44.1kHz)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, 
        orig_freq=sample_rate, 
        new_freq=generator.sample_rate
    )
    return audio_tensor.squeeze(0) # Trả về tensor 1D

# 2. Tạo danh sách context segments theo định dạng CSM yêu cầu
segments = []
for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths):
    segments.append({
        "text": transcript,
        "speaker": speaker,
        "audio": load_audio(audio_path).to(device)
    })

# 3. Generate audio mới dựa trên ngữ cảnh (In-context learning)
# Lưu ý: generator.generate trả về tensor âm thanh và sample rate
output_audio = generator.generate(
    text="me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10000,
)

# 4. Lưu file kết quả
# Đảm bảo đưa tensor về CPU trước khi save
torchaudio.save(
    "audio_output.wav", 
    output_audio.unsqueeze(0).cpu(), 
    generator.sample_rate
)

print("Đã tạo file audio_output.wav thành công!")
