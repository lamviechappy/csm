import os
from pathlib import Path
import torch

# Giả sử các module của bạn nằm trong cùng thư mục
from studio_engine import CSMStudioEngine, PodcastProject
import config

def main():
    # Gọi hàm setup để đảm bảo thư mục CSM_Projects tồn tại
    config.setup_default_workspace()

    # Chỉ cần truyền tên dự án
    project_name = "Test_Project_2026"
    
    try:
        # Engine sẽ tự hiểu là nằm trong ~/CSM_Projects/Test_Project_2026
        engine = CSMStudioEngine(project_name=project_name)
    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        return


    # 3. Chạy phần [WARMUP]
    print("\n🔥 Step 1: Building global warmup...")
    # Gọi qua CSMStudioEngine, nó sẽ delegate xuống PodcastGeneratorEngine
    warmup_segments = engine.prepare_warmup()

    # Kiểm tra an toàn để tránh lỗi TypeError: object of type 'NoneType' has no len()
    if warmup_segments is None:
        print("[FAIL] Warmup failed: Function returned None")
        warmup_segments = [] # Gán list rỗng để các bước sau không bị crash
    else:
        print(f"[SUCCESS] Warmup ready: {len(warmup_segments)} segments created.")

    # 4. Giả lập kịch bản (Script) để tạo Podcast
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

    # 5. Chạy phần [GENERATION]
    print("\n🎙️ Step 2: Generating podcast segments...")
    try:
        # Gọi hàm generate của CSMStudioEngine (hàm này gọi self.engine.generate_podcast)
        final_segments = engine.generate(test_script)
        
        if final_segments:
            print(f"[SUCCESS] Generated {len(final_segments)} final audio segments.")
            
            # Kiểm tra xem file đã được lưu thực tế chưa
            output_dir = engine.project.get_audio_output_dir()
            print(f"[TEST] Check your output folder: {output_dir}")
        else:
            print("[WARNING] Generation completed but no segments were returned.")
            
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")

    print("\n" + "="*40)
    print("✅ TEST COMPLETED")
    print("="*40)

if __name__ == "__main__":
    # Đảm bảo môi trường chạy đúng thiết bị (CPU/GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SYSTEM] Running on device: {device}")
    
    main()
