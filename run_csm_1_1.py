import os
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from generator import load_csm_1b, Segment
from dataclasses import dataclass

# 1. THIẾT LẬP MÔI TRƯỜNG (Đặt ở đầu file để có hiệu lực)
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Ép buộc fallback CPU cho các toán tử chưa được MPS hỗ trợ (như istft)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

prompt_filepath_conversational_a = "/Volumes/SSD256/dev-projects/csm/voice_samples/woman-1.wav"
prompt_filepath_conversational_b = "/Volumes/SSD256/dev-projects/csm/voice_samples/man-1.wav"

SPEAKER_PROMPTS = {
    "conversational_a": {
        "text": "And our rule is the same as always. Real conversation first, vocabulary, hidden inside, no boring lists while you're trying to",
        "audio": prompt_filepath_conversational_a
    },
    "conversational_b": {
        "text": "And just so you know, this episode is around B1 level. So the English is simple, slow, and natural. And then you can really use in real",
        "audio": prompt_filepath_conversational_b
    }
}

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    # Lấy channel đầu tiên nếu là stereo
    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor[0]
    else:
        audio_tensor = audio_tensor.squeeze(0)
    
    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)

def main():
    # 2. CHỌN DEVICE: Ép về CPU cho Mac Silicon để tránh lỗi istft/float64
    # Dù M4 mạnh nhưng bản PyTorch này yêu cầu Triton/CUDA để chạy GPU hiệu quả.
    device = "cpu"
    print(f"Using device: {device}")

    # 3. LOAD MODEL
    generator = load_csm_1b(device)

    # 4. CHUẨN BỊ PROMPTS
    print("Preparing speaker prompts...")
    prompt_a = prepare_prompt(
        SPEAKER_PROMPTS["conversational_a"]["text"],
        0,
        SPEAKER_PROMPTS["conversational_a"]["audio"],
        generator.sample_rate
    )

    prompt_b = prepare_prompt(
        SPEAKER_PROMPTS["conversational_b"]["text"],
        1,
        SPEAKER_PROMPTS["conversational_b"]["audio"],
        generator.sample_rate
    )

    # 5. DANH SÁCH HỘI THOẠI
    # conversation = [
    #     {"text": "Thank you for calling support. My name is Alex. How can I help you today?", "speaker_id": 1},
    #     {"text": "Hi Alex, I'm having some trouble with my account login. It keeps saying 'Invalid Credentials' even after a reset.", "speaker_id": 0},
    #     {"text": "I'm sorry to hear that. Let's try to fix it. Are you using a VPN or any browser extensions that might block cookies?", "speaker_id": 1},
    #     {"text": "Now that you mention it, I do have a new ad-blocker installed. Could that be the issue?", "speaker_id": 0},
    #     {"text": "It's very likely. Could you try disabling it for a moment and refreshing the page?", "speaker_id": 1},
    #     {"text": "Wow, that actually worked! I'm back in. Thanks for the quick fix, Alex.", "speaker_id": 0},
    #     {"text": "You're very welcome! Is there anything else I can assist you with before you go?", "speaker_id": 1},
    #     {"text": "No, that was the only thing. Have a great day!", "speaker_id": 0}
    # ]

    conversation =[
            {
                    "text": "Hello, everyone, and welcome back to Easy English Together. We're learning English is easy and fun. I'm Emily.",
                    "speaker_id": 0
            },
            {
                    "text": "Hello, Emily. Hello, everyone. It's great to be here.",
                    "speaker_id": 1
            },
            {
                    "text": "Hi, Mark. How are you today?",
                    "speaker_id": 0
            },
            {
                    "text": "I'm doing great. Thank you. And you, you look very happy today.",
                    "speaker_id": 1
            },
            {
                    "text": "I'm very happy. I'm excited for our topic today.",
                    "speaker_id": 0
            },
            {
                    "text": "Oh, yes. It's a very good topic. Very important.",
                    "speaker_id": 1
            },
            {
                    "text": "Yes, exactly. But before we start everyone, please remember to subscribe to our channel.",
                    "speaker_id": 0
            },
            {
                    "text": "Yes, and click the like button and please share our podcast with your friends and family. It helps us a lot.",
                    "speaker_id": 1
            },
            {
                    "text": "It really does. Okay, so Mark, are you ready?",
                    "speaker_id": 0
            },
            {
                    "text": "I am ready. So, what is our topic today?",
                    "speaker_id": 1
            },
            {
                    "text": "Today, our topic is tell me about yourself.",
                    "speaker_id": 0
            },
            {
                    "text": "Ah, a very common question. People ask this all the time.",
                    "speaker_id": 1
            },
            {
                    "text": "They do in a new class, at a new job, when you meet new people.",
                    "speaker_id": 0
            },
            {
                    "text": "It's true. And sometimes it's hard to know what to say.",
                    "speaker_id": 1
            },
            {
                    "text": "Yes, so today we will talk about it. We will make it easy and fun.",
                    "speaker_id": 0
            },
            {
                    "text": "That sounds perfect. So, who starts you or me?",
                    "speaker_id": 1
            },
            {
                    "text": "Hmm. How about you start Mark? Tell me about yourself.",
                    "speaker_id": 0
            },
            {
                    "text": "Okay, okay. My turn first. Where do I start?",
                    "speaker_id": 1
            },
            {
                    "text": "Let's start with the easy one. What is your name?",
                    "speaker_id": 0
            }
    ]
    generated_segments = []
    prompt_segments = [prompt_a, prompt_b]

    # 6. VÒNG LẶP SINH AUDIO
    for i, utterance in enumerate(conversation):
        print(f"Generating ({i+1}/{len(conversation)}): {utterance['text']}")
        try:
            # Inference mode giúp tiết kiệm RAM M4
            with torch.inference_mode():
                audio_tensor = generator.generate(
                    text=utterance['text'],
                    speaker=utterance['speaker_id'],
                    context=prompt_segments + generated_segments,
                    max_audio_length_ms=10_000,
                )
            
            if audio_tensor is not None:
                # Đảm bảo audio_tensor ở dạng CPU float32 để torch.cat không lỗi
                audio_tensor = audio_tensor.cpu().float()
                generated_segments.append(Segment(text=utterance['text'], speaker=utterance['speaker_id'], audio=audio_tensor))
            else:
                print(f"⚠️ Warning: Sentence {i+1} returned None.")
        except Exception as e:
            print(f"❌ Error at sentence {i+1}: {str(e)}")

    # 7. KẾT NỐI VÀ LƯU FILE
    if len(generated_segments) > 0:
        print(f"Concatenating {len(generated_segments)} segments...")
        # Đảm bảo list các tensor không rỗng trước khi cat
        all_audio = torch.cat([seg.audio for seg in generated_segments], dim=0)
        
        output_filename = "full_conversation.wav"
        torchaudio.save(
            output_filename,
            all_audio.unsqueeze(0),
            generator.sample_rate
        )
        print(f"✅ Successfully generated {output_filename}")
    else:
        print("❌ Error: No segments were generated. Check the errors above.")

if __name__ == "__main__":
    main()
