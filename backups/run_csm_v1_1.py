import os
import time
import torch
import torchaudio
import gradio as gr
from datetime import datetime

# --------- ENV (MAC SAFE) ----------
os.environ["NO_TORCH_COMPILE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --------- LOCAL IMPORT ----------
from generator import load_csm_1b, Segment
from utils import parse_conversation   # <-- bạn đã đưa hàm vào đây
import config


# ===============================
#  SPEAKER PROMPTS (GIỮ NGUYÊN CHẤT LƯỢNG GỐC)
# ===============================

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


# ===============================
#  AUDIO UTILS
# ===============================

def load_prompt_audio(audio_path: str, target_sample_rate: int) -> torch.Tensor:
    audio_tensor, sample_rate = torchaudio.load(audio_path)

    if audio_tensor.shape[0] > 1:
        audio_tensor = audio_tensor[0]
    else:
        audio_tensor = audio_tensor.squeeze(0)

    audio_tensor = torchaudio.functional.resample(
        audio_tensor, orig_freq=sample_rate, new_freq=target_sample_rate
    )
    return audio_tensor

import glob

VOICE_SAMPLE_DIR = "/Volumes/SSD256/dev-projects/csm/voice_samples"

def scan_voice_samples():
    wavs = glob.glob(os.path.join(VOICE_SAMPLE_DIR, "*.wav"))
    pairs = {}
    for w in wavs:
        name = os.path.splitext(os.path.basename(w))[0]
        txt = os.path.join(VOICE_SAMPLE_DIR, name + ".txt")
        if os.path.exists(txt):
            pairs[name] = {"wav": w, "txt": txt}
    return pairs

VOICE_PAIRS = scan_voice_samples()
VOICE_NAMES = list(VOICE_PAIRS.keys())


def prepare_prompt_from_pair(pair_name: str, speaker: int, sample_rate: int) -> Segment:
    wav = VOICE_PAIRS[pair_name]["wav"]
    txt = VOICE_PAIRS[pair_name]["txt"]

    with open(txt, "r", encoding="utf-8") as f:
        text = f.read().strip()

    audio_tensor = load_prompt_audio(wav, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)


def prepare_prompt(text: str, speaker: int, audio_path: str, sample_rate: int) -> Segment:
    audio_tensor = load_prompt_audio(audio_path, sample_rate)
    return Segment(text=text, speaker=speaker, audio=audio_tensor)


# ===============================
#  LOAD MODEL
# ===============================

def load_generator():
    if config.RUNNING_MODE == "Offline":
        os.environ["HF_HOME"] = "/Volumes/SSD256/ai-models"
        print("[CSM] Offline mode enabled")

    print("[CSM] Loading model...")
    device = config.DEVICE
    gen = load_csm_1b(device)
    print("[CSM] Model loaded on", device)
    return gen


generator = load_generator()


# ===============================
#  CORE PIPELINE (CHẤT LƯỢNG GỐC)
# ===============================

def run_conversation(speaker0_voice, speaker1_voice, conversation_text, prefix, merge_audio, love_my_mac, every_x, sleep_y):

    conversation = parse_conversation(conversation_text)
    # conversation = [
    #         {"text": "Thank you for calling support. My name is Alex. How can I help you today?", "speaker_id": 1},
    #         {"text": "Hi Alex, I'm having some trouble with my account login. It keeps saying 'Invalid Credentials' even after a reset.", "speaker_id": 0},
    #         {"text": "I'm sorry to hear that. Let's try to fix it. Are you using a VPN or any browser extensions that might block cookies?", "speaker_id": 1},
    #         {"text": "Now that you mention it, I do have a new ad-blocker installed. Could that be the issue?", "speaker_id": 0},
    #         {"text": "It's very likely. Could you try disabling it for a moment and refreshing the page?", "speaker_id": 1},
    #         {"text": "Wow, that actually worked! I'm back in. Thanks for the quick fix, Alex.", "speaker_id": 0},
    #         {"text": "You're very welcome! Is there anything else I can assist you with before you go?", "speaker_id": 1},
    #         {"text": "No, that was the only thing. Have a great day!", "speaker_id": 0}
    #     ]
    if len(conversation) == 0:
        return "❌ No valid conversation lines found.", None
    print(f"Conversation = {conversation}")
    print(f"[CSM] Parsed {len(conversation)} turns")

    # --- prepare prompts (GIỐNG BẢN GỐC) ---
    prompt_a = prepare_prompt_from_pair(speaker0_voice, 0, generator.sample_rate)
    prompt_b = prepare_prompt_from_pair(speaker1_voice, 1, generator.sample_rate)
    prompt_segments = [prompt_a, prompt_b]
    # generated_segments = []

    outputs = []
    # counter = 0

    # =========================
    # 6. VÒNG LẶP SINH AUDIO (GIỮ LOGIC GỐC)
    # =========================

    generated_segments = []

    for i, utterance in enumerate(conversation):
        print(f"Generating ({i+1}/{len(conversation)}): {utterance['text']}")

        try:
            # Inference mode giúp tiết kiệm RAM + ổn định
            with torch.inference_mode():
                audio_tensor = generator.generate(
                    text=utterance["text"],
                    speaker=utterance["speaker_id"],
                    context=prompt_segments + generated_segments,
                    max_audio_length_ms=10_000,
                )

            if audio_tensor is not None:
                # RẤT QUAN TRỌNG: đảm bảo đúng dtype & device
                audio_tensor = audio_tensor.cpu().float()

                generated_segments.append(
                    Segment(
                        text=utterance["text"],
                        speaker=utterance["speaker_id"],
                        audio=audio_tensor
                    )
                )
            else:
                print(f"⚠️ Warning: Sentence {i+1} returned None.")

        except Exception as e:
            print(f"❌ Error at sentence {i+1}: {str(e)}")


        merged_path = None
        if merge_audio and len(generated_segments) > 1:
            print("[CSM] Merging audio...")
            all_audio = torch.cat([s.audio for s in generated_segments], dim=0)
            ts = datetime.now().strftime("%d%H%M%S")
            merged_name = f"{prefix}_FULL_{ts}.wav"
            merged_path = os.path.join(config.OUTPUT_DIR, merged_name)
            torchaudio.save(merged_path, all_audio.unsqueeze(0), generator.sample_rate)

        return "✅ Done", merged_path if merged_path else outputs


# ===============================
#  GRADIO UI
# ===============================

with gr.Blocks() as demo:
    gr.Markdown("# 🎙 CSM Conversation Generator – v1.1 (Quality Mode)")

    gr.Markdown("Conversation format example:\n"
                "[SPEAKER_00] Hello...\n"
                "[SPEAKER_01] Hi...\n")

    conversation_box = gr.Textbox(lines=14, label="Conversation")
    with gr.Row():
        speaker0_voice = gr.Dropdown(VOICE_NAMES, label="Speaker 0 voice")
        speaker1_voice = gr.Dropdown(VOICE_NAMES, label="Speaker 1 voice")

    with gr.Row():
        prefix = gr.Textbox(value="pod01", label="Output prefix")
        merge_audio = gr.Checkbox(value=True, label="Merge all audio")

    with gr.Accordion("🍏 Love My Mac Mode", open=False):
        love_my_mac = gr.Checkbox(value=False, label="Enable")
        every_x = gr.Number(value=config.LOVE_MY_MAC_EVERY_X, precision=0, label="Every X segments")
        sleep_y = gr.Number(value=config.LOVE_MY_MAC_SLEEP_Y, precision=0, label="Sleep Y seconds")

    run_btn = gr.Button("🚀 Generate")
    status = gr.Textbox(label="Status")
    output_audio = gr.File(label="Generated audio")

    run_btn.click(
        run_conversation,
        inputs=[speaker0_voice, speaker1_voice, conversation_box, prefix, merge_audio, love_my_mac, every_x, sleep_y]
,
        outputs=[status, output_audio]
    )
    

demo.launch(server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT)
