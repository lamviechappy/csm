import os
import torch
import torchaudio
import numpy as np
from generator import load_csm_1b, Segment

# 0. VÔ HIỆU HOÁ CẢNH BÁO
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 1. VÔ HIỆU HÓA HOÀN TOÀN TRITON & COMPILATION
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Cho phép chạy CPU nếu MPS chưa hỗ trợ 1 số hàm
prompt_filepath_conversational_a = "/Volumes/SSD256/dev-projects/csm/voice_samples/woman-1.wav"
prompt_filepath_conversational_b = "/Volumes/SSD256/dev-projects/csm/voice_samples/man-1.wav"
SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": (
            "And our rule is the same as always."
            "Real conversation first, vocabulary, hidden inside,"
            "no boring lists while you're trying to"
        ),
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": (
            "And just so you know, this episode is around B1 level."
            "So the English is simple, slow, and natural."
            "And then you can really use in real"
        ),
        "audio": prompt_filepath_conversational_b
    }
}
def main():
    # 2. TỐI ƯU HÓA DEVICE CHO M4
    # Mặc dù CSM có thể dùng float64, nhưng chip M4 xử lý cực nhanh với MPS (Metal)
    # Chúng ta ưu tiên MPS cho tốc độ, nếu lỗi sẽ fallback về CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Using Device: Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("ℹ️ Using Device: CPU")

    # 3. LOAD MODEL VỚI DEVICE ĐÃ CHỌN
        # Load model
    generator = load_csm_1b(device)

    # Sửa lỗi AttributeError: Sử dụng _model thay vì model
    if hasattr(generator, '_model'):
        generator._model = generator._model.to(device).to(torch.float32)
    elif hasattr(generator, 'model'):
        generator.model = generator.model.to(device).to(torch.float32)


    # ... (giữ nguyên phần SPEAKER_PROMPTS và load_prompt_audio từ code của bạn) ...

    # 4. CHUẨN BỊ PROMPT VÀ ÉP KIỂU
    def prepare_prompt_optimized(text, speaker, audio_path, sample_rate):
        audio_tensor, sr = torchaudio.load(audio_path)
        audio_tensor = audio_tensor.mean(0) # Convert to mono
        if sr != sample_rate:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, sample_rate)
        
        # Đưa audio lên cùng device và kiểu dữ liệu với model
        return Segment(text=text, speaker=speaker, audio=audio_tensor.to(device).to(torch.float32))

    prompt_a = prepare_prompt_optimized(
        SPEAKER_PROMPTS["conversational_a"]["text"], 0,
        SPEAKER_PROMPTS["conversational_a"]["audio"], generator.sample_rate
    )
    prompt_b = prepare_prompt_optimized(
        SPEAKER_PROMPTS["conversational_b"]["text"], 1,
        SPEAKER_PROMPTS["conversational_b"]["audio"], generator.sample_rate
    )

    # 5. GENERATION LOOP (VỚI TỐI ƯU BỘ NHỚ)
    # conversation = [
    #     {"text": "Hey how are you doing?", "speaker_id": 0},
    #     {"text": "Pretty good, pretty good. How about you?", "speaker_id": 1},
    # ]

    conversation = [
        {"text": "Thank you for calling support. My name is Alex. How can I help you today?", "speaker_id": 1},
        {"text": "Hi Alex, I'm having some trouble with my account login. It keeps saying 'Invalid Credentials' even after a reset.", "speaker_id": 0},
        # {"text": "I'm sorry to hear that. Let's try to fix it. Are you using a VPN or any browser extensions that might block cookies?", "speaker_id": 1},
        # {"text": "Now that you mention it, I do have a new ad-blocker installed. Could that be the issue?", "speaker_id": 0},
        # {"text": "It's very likely. Could you try disabling it for a moment and refreshing the page?", "speaker_id": 1},
        # {"text": "Wow, that actually worked! I'm back in. Thanks for the quick fix, Alex.", "speaker_id": 0},
        # {"text": "You're very welcome! Is there anything else I can assist you with before you go?", "speaker_id": 1},
        # {"text": "No, that was the only thing. Have a great day!", "speaker_id": 0}
    ]

    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    for utterance in conversation:
        print(f"🎙️ Generating: {utterance['text']}")
        
        # Trước khi gọi generate, hãy tạm thời ép tensor về cpu nếu nó gây lỗi istft
        with torch.inference_mode():
            # Đảm bảo các prompt/context cũng được xử lý đồng nhất
            audio_tensor = generator.generate(
                text=utterance['text'],
                speaker=utterance['speaker_id'],
                context=prompt_segments + generated_segments,
                max_audio_length_ms=10_000,
            )
            # Chuyển kết quả về CPU ngay lập tức để tránh lỗi logic istft ở các bước sau
            audio_tensor = audio_tensor.cpu() 


    # 6. KẾT XUẤT FILE
    # all_audio = torch.cat([seg.audio.cpu() for seg in generated_segments], dim=0)
    # torchaudio.save("full_conversation_m4.wav", all_audio.unsqueeze(0), generator.sample_rate)
    # print("✅ Done! Saved to full_conversation_m4.wav")
    # Kiểm tra xem có dữ liệu không trước khi nối
    if not generated_segments:
        print("❌ Error: No audio was generated. Please check the conversation loop.")
        return

    print(f"Concatenating {len(generated_segments)} segments...")
    
    # Đảm bảo tất cả audio tensor đều ở trên CPU và đúng định dạng
    tensors_to_cat = []
    for seg in generated_segments:
        if seg.audio is not None:
            # Chuyển về CPU và loại bỏ các chiều dư thừa (squeeze)
            tensors_to_cat.append(seg.audio.cpu().squeeze())

    all_audio = torch.cat(tensors_to_cat, dim=0)
    
    # Lưu file
    output_path = "full_conversation_m4.wav"
    torchaudio.save(
        output_path,
        all_audio.unsqueeze(0), # Thêm lại chiều channel [1, T]
        generator.sample_rate
    )
    print(f"✅ Successfully generated {output_path}")


if __name__ == "__main__":
    main()
