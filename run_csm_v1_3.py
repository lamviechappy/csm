import os
import re
import time
import glob
import torch
import torchaudio
import gradio as gr
from datetime import datetime

from generator import load_csm_1b, Segment
from utils_v1_3 import parse_conversation, load_prompt_segment


# =========================
# ENV (giữ như app gốc)
# =========================
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

DEVICE = "cpu"   # Mac mini M4 → giữ CPU để ổn định


# =========================
# PATHS
# =========================
VOICE_DIR = "/Volumes/SSD256/dev-projects/csm/voice_samples"
OUTPUT_DIR = "/Volumes/SSD256/dev-projects/csm/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


from config import WARMUP_PRESETS, MAX_HISTORY_SEGMENTS


# =========================
# LOAD SPEAKERS
# =========================
def load_speakers():
    speakers = {}
    wavs = glob.glob(os.path.join(VOICE_DIR, "*.wav"))

    for wav in wavs:
        name = os.path.splitext(os.path.basename(wav))[0]
        txt = os.path.join(VOICE_DIR, name + ".txt")
        if os.path.exists(txt):
            speakers[name] = {
                "audio": wav,
                "text": txt
            }
    return speakers


SPEAKERS = load_speakers()
SPEAKER_NAMES = list(SPEAKERS.keys())


# =========================
# CORE ENGINE
# =========================
def run_conversation(
    speaker_a_name,
    speaker_b_name,
    conversation_text,

    use_warmup,
    warmup_preset_name,
    warmup_custom_text,

    file_prefix,
    merge_audio,
    love_my_mac,
    pause_every,
    pause_seconds,
):
    import os, time, torch
    from utils_v1_3 import parse_conversation
    from generator import Segment

    print("\n[CSM] ===== RUN CONVERSATION v1.3 =====")


    print("Speaker A:", speaker_a_name)
    print("Speaker B:", speaker_b_name)

    generator = load_csm_1b(DEVICE)


    # ------------------------
    # 0. Resolve speakers
    # ------------------------
    speaker_a = SPEAKERS[speaker_a_name]
    speaker_b = SPEAKERS[speaker_b_name]

    # SPEAKER_PROMPTS = {
    #     0: speaker_a,
    #     1: speaker_b,
    # }

    # ------------------------
    # 1. Parse main conversation
    # ------------------------
    conversation = parse_conversation(conversation_text)

    if not conversation:
        return "❌ No valid conversation found. Please use [SPEAKER_00] ... format."

    print(f"[CSM] Parsed {len(conversation)} main turns")

    # ------------------------
    # 2. Build prompt segments
    # ------------------------
    prompt_segments = []
    generated_segments = []

    # def load_prompt_segment(text, wav_path, speaker_id):
    #     audio = load_audio(wav_path, target_sr=generator.sample_rate)
    #     return Segment(text=text, speaker=speaker_id, audio=audio)

    # ----- load prompt segments (GIỐNG APP GỐC) -----
    prompt_segments = [
        load_prompt_segment(speaker_a["text"], speaker_a["audio"], 0, generator.sample_rate),
        load_prompt_segment(speaker_b["text"], speaker_b["audio"], 1, generator.sample_rate),
    ]    
    
    
    
    # prompt_segments.append(
    #     load_prompt_segment(speaker_a["text"], speaker_a["audio"], 0)
    # )
    # prompt_segments.append(
    #     load_prompt_segment(speaker_b["text"], speaker_b["audio"], 1)
    # )

    generated_segments.extend(prompt_segments)




    print("[CSM] Loaded speaker prompts")

    # ------------------------
    # 3. WARMUP (optional)
    # ------------------------
    
    from config import MAX_HISTORY_SEGMENTS
    if use_warmup:
        print("[CSM] Warmup enabled")

        if warmup_custom_text.strip():
            warmup_text = warmup_custom_text
            print("[CSM] Using custom warmup")
        else:
            warmup_text = WARMUP_PRESETS.get(warmup_preset_name, "")
            print(f"[CSM] Using warmup preset: {warmup_preset_name}")

        warmup_conversation = parse_conversation(warmup_text)
        print(f"[CSM] Warmup turns: {len(warmup_conversation)}")

        for i, utt in enumerate(warmup_conversation):
            print(f"[WARMUP {i+1}/{len(warmup_conversation)}] {utt['text']}")

            try:
                with torch.inference_mode():
                    audio_tensor = generator.generate(
                        text=utt["text"],
                        speaker=utt["speaker_id"],
                        context=generated_segments[-MAX_HISTORY_SEGMENTS:],
                        max_audio_length_ms=6000,
                    )

                if audio_tensor is None:
                    continue

                audio_tensor = audio_tensor.cpu().float()
                seg = Segment(
                    text=utt["text"],
                    speaker=utt["speaker_id"],
                    audio=audio_tensor
                )
                generated_segments.append(seg)

            except Exception as e:
                print(f"❌ Warmup error: {e}")

    # ------------------------
    # 4. MAIN conversation
    # ------------------------
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_files = []

    for i, utt in enumerate(conversation):
        print(f"[CSM] Generating ({i+1}/{len(conversation)}): {utt['text']}")

        try:
            context_data = generated_segments[-MAX_HISTORY_SEGMENTS:]
            
            with torch.inference_mode():
                audio_tensor = generator.generate(
                    text=utt["text"],
                    speaker=utt["speaker_id"],
                    context=generated_segments[-MAX_HISTORY_SEGMENTS:],
                    max_audio_length_ms=10_000,
                )
            # In ra để kiểm tra 
            print(f"Context: {context_data}")

            if audio_tensor is None:
                print("⚠️ Returned None")
                continue

            audio_tensor = audio_tensor.cpu().float()

            seg = Segment(
                text=utt["text"],
                speaker=utt["speaker_id"],
                audio=audio_tensor
            )

            generated_segments.append(seg)

            # ---------- save ----------
            ts = datetime.now().strftime("%d%H%M%S")
            filename = f"{file_prefix}_{i+1:02d}_{ts}.wav"
            # filename = f"{file_prefix}_{time.strftime('%d%H%M%S')}_{i+1:02d}.wav"
            out_path = os.path.join(OUTPUT_DIR, filename)

            torchaudio.save(out_path, audio_tensor.unsqueeze(0), generator.sample_rate)
            
            output_files.append(out_path)

            # # ---------- Love my Mac ----------
            # if love_my_mac and (i + 1) % pause_every == 0:
            #     print(f"[CSM] Cooling Mac for {pause_seconds}s...")
            #     torch.mps.empty_cache()
            #     time.sleep(pause_seconds)

            # ----- Love my Mac mode -----
            if love_my_mac and (i + 1) % pause_every == 0:
                print(f"💤 Cooling down {pause_seconds}s ...")
                torch.mps.empty_cache() if torch.backends.mps.is_available() else None
                time.sleep(pause_seconds)


        except Exception as e:
            print(f"❌ Error at sentence {i+1}: {e}")

    # ------------------------
    # 5. Merge if needed
    # ------------------------
    # if merge_audio and output_files:
    #     merged_path = os.path.join(
    #         OUTPUT_DIR, f"{file_prefix}_MERGED_{time.strftime('%d%H%M%S')}.wav"
    #     )
    #     merge_wavs(output_files, merged_path)
    #     return merged_path

    # return output_files

    import os
    merged_path = None

    if merge_audio and output_files:    
        all_audio = torch.cat([s.audio for s in generated_segments], dim=0)
        ts = datetime.now().strftime("%d%H%M%S")
        merged_path = os.path.join(OUTPUT_DIR, f"{file_prefix}_FULL_{ts}.wav")
        torchaudio.save(merged_path, all_audio.unsqueeze(0), generator.sample_rate)

    return output_files, merged_path


with gr.Blocks(title="CSM Podcast Generator v1.3") as demo:

    gr.Markdown("## 🎙️ CSM Podcast Generator – v1.3 (Mac Optimized)")

    # =========================
    # SPEAKER SETUP
    # =========================
    with gr.Row():
        speaker_a_dropdown = gr.Dropdown(
            choices=list(SPEAKERS.keys()),
            value=list(SPEAKERS.keys())[0],
            label="Speaker A (SPEAKER_00)"
        )

        speaker_b_dropdown = gr.Dropdown(
            choices=list(SPEAKERS.keys()),
            value=list(SPEAKERS.keys())[1] if len(SPEAKERS) > 1 else list(SPEAKERS.keys())[0],
            label="Speaker B (SPEAKER_01)"
        )

    # =========================
    # WARMUP SECTION
    # =========================
    gr.Markdown("## 🔥 Warmup (Build emotion & speaking style)")

    use_warmup_checkbox = gr.Checkbox(value=True, label="Use warmup lines")

    with gr.Row():
        warmup_preset_dropdown = gr.Dropdown(
            choices=list(WARMUP_PRESETS.keys()),
            value=list(WARMUP_PRESETS.keys())[0],
            label="Warmup preset"
        )

    warmup_custom_textbox = gr.Textbox(
        lines=6,
        placeholder="[SPEAKER_00] ...\n[SPEAKER_01] ...",
        label="Or custom warmup (optional)"
    )

    # =========================
    # MAIN CONVERSATION
    # =========================
    gr.Markdown("## 💬 Main Conversation")

    conversation_box = gr.Textbox(
        lines=14,
        placeholder="[SPEAKER_00] Hello...\n[SPEAKER_01] Hi...",
        label="Conversation input"
    )

    # =========================
    # OUTPUT SETTINGS
    # =========================
    gr.Markdown("## 💾 Output settings")

    with gr.Row():
        file_prefix_box = gr.Textbox(value="pod01", label="Filename prefix")

        merge_checkbox = gr.Checkbox(value=True, label="Merge audio after generation")

    # =========================
    # LOVE MY MAC
    # =========================
    gr.Markdown("## 🍎 Love my Mac")

    love_mac_checkbox = gr.Checkbox(value=True, label="Enable cooling mode")

    with gr.Row():
        pause_every_slider = gr.Slider(1, 10, value=3, step=1, label="Pause every N sentences")
        pause_seconds_slider = gr.Slider(1, 20, value=5, step=1, label="Pause seconds")

    # =========================
    # RUN
    # =========================
    run_btn = gr.Button("🚀 Generate podcast")

    output_audio = gr.File(label="Generated audio")

    run_btn.click(
        fn=run_conversation,
        inputs=[
            speaker_a_dropdown,
            speaker_b_dropdown,

            conversation_box,

            use_warmup_checkbox,
            warmup_preset_dropdown,
            warmup_custom_textbox,

            file_prefix_box,
            merge_checkbox,
            love_mac_checkbox,
            pause_every_slider,
            pause_seconds_slider,
        ],
        outputs=output_audio
    )

demo.launch(server_port=7863, share=False)
