# mini_studio_gui_v2.py
# ==========================================
# 🎙 MINI STUDIO GUI V2 – PRODUCTION VOICE LAB
# Author: built with ChatGPT for Donald
# Focus: stability, voice consistency, UX, mac safety
# ==========================================

import os
import re
import json
import time
import gc
import threading
import subprocess
from datetime import datetime
from pathlib import Path

import torch
import torchaudio
import gradio as gr

# Tokenizers chạy đơn luồng để tránh xung đột.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# ⚠️ IMPORT CSM CORE
# =========================
from generator import load_csm_1b
from generator import Segment


# =========================
# 🎯 GLOBAL CONFIG
# =========================
APP_DIR = Path(__file__).parent
VOICE_SAMPLE_DIR = APP_DIR / "voice_samples"
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"

SESSION_FILE = "session.json"
NOTES_FILE = "notes.md"

STOP_REQUESTED = False
GENERATOR = None


# =========================
# 🧠 UTILS
# =========================
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def open_folder(path: str):
    try:
        if os.name == "posix":
            subprocess.Popen(["open", path])
        elif os.name == "nt":
            os.startfile(path)
    except Exception as e:
        print("Open folder error:", e)


def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================
# 🎤 AUDIO UTILS
# =========================
def load_audio(wav_path, target_sr):
    wav_path_str = str(wav_path) 
    wav, sr = torchaudio.load(wav_path_str) 
    # wav, sr = torchaudio.load(wav_path)
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


# =========================
# 🎧 VOICE SAMPLE SCAN
# Only return voices that have BOTH wav + txt
# =========================
def scan_voice_samples():
    if not os.path.exists(VOICE_SAMPLE_DIR):
        return []

    files = os.listdir(VOICE_SAMPLE_DIR)
    wavs = {os.path.splitext(f)[0] for f in files if f.endswith(".wav")}
    txts = {os.path.splitext(f)[0] for f in files if f.endswith(".txt")}

    valid = sorted(list(wavs & txts))
    return valid


# =========================
# 🔢 ORDER NUMBER TOOLS
# =========================
def insert_order_no(text):
    lines = []
    idx = 1
    for line in text.splitlines():
        if line.strip():
            if not re.match(r"\[\d+\]", line):
                line = f"[{idx:03d}] {line}"
                idx += 1
        lines.append(line)
    return "\n".join(lines)


def delete_order_no(text):
    return "\n".join([re.sub(r"^\[\d+\]\s*", "", l) for l in text.splitlines()])


# =========================
# 🎬 EPISODE ENVIRONMENT
# =========================

EPISODE_FILE = "episode.json"
NOTES_FILE = "notes.md"


def default_notes():
    return f"""# 🎙 Mini Studio – Episode Notes

## 📅 Episode Info
- Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Project name:
- Description:

## 🎤 Voices
- Speaker 0:
- Speaker 1:

## 🧱 Blocks
- Good:
- Need regen:
- Notes:

## 🧠 General Notes
- 
"""


def default_episode_state(episode_dir: Path):
    return {
        "episode_dir": str(episode_dir),
        "created_at": datetime.now().isoformat(),
        "last_session": None,

        "speakers": {
            "speaker_0": "",
            "speaker_1": ""
        },

        "block_size": 5,

        "love_my_mac": {
            "enabled": False,
            "cool_every": 5,
            "cool_seconds": 10
        }
    }


def save_episode(episode_dir: Path, data: dict):
    path = episode_dir / EPISODE_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_episode(episode_dir: Path):
    path = episode_dir / EPISODE_FILE
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_notes(episode_dir: Path, content: str):
    path = episode_dir / NOTES_FILE
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# Trong phần 1, sửa lại hàm để linh hoạt hơn:
def save_notes_ui(content: str, episode_path: str):
    if not episode_path: return "❌ Vui lòng nhập đường dẫn Output"
    save_notes(Path(episode_path), content)
    return f"✅ Đã lưu ghi chú lúc {datetime.now().strftime('%H:%M:%S')}"

def load_notes(episode_dir: Path):
    path = episode_dir / NOTES_FILE
    if not path.exists():
        content = default_notes()
        save_notes(episode_dir, content)
        return content
    return path.read_text(encoding="utf-8")


def ensure_episode_environment(episode_dir: Path):
    """
    Ensure that an episode folder is a valid Mini Studio episode.

    It will:
    - create folder if not exists
    - create notes.md if missing
    - create episode.json if missing
    - load both if existing

    Return:
        episode_state (dict)
        notes_content (str)
    """

    episode_dir.mkdir(parents=True, exist_ok=True)

    # ---- Episode state
    state = load_episode(episode_dir)
    if state is None:
        state = default_episode_state(episode_dir)
        save_episode(episode_dir, state)

    # ---- Notes
    notes = load_notes(episode_dir)

    return state, notes



# =========================
# 🧠 SESSION
# =========================
def save_session(output_dir, state: dict):
    path = Path(output_dir) / SESSION_FILE
    save_json(path, state)


def load_session(output_dir):
    path = Path(output_dir) / SESSION_FILE
    return load_json(path)



# =========================
# ⚙️ ENGINE CORE
# =========================

def request_stop():
    global STOP_REQUESTED
    STOP_REQUESTED = True
    return "🛑 Stop requested. Finishing current step..."

def exit_app():
    """
    Gracefully stop generation, release memory, and terminate the app process.
    This is a REAL exit (not just closing the browser tab).
    """

    global STOP_REQUESTED
    STOP_REQUESTED = True

    print("\n[EXIT] Exit app requested...")
    time.sleep(0.3)

    # ---- Clear PyTorch cache
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[EXIT] CUDA cache cleared")

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
            print("[EXIT] MPS cache cleared")
    except Exception as e:
        print("[EXIT] Cache clear warning:", e)

    # ---- Force Python garbage collection
    try:
        gc.collect()
        print("[EXIT] Garbage collection done")
    except:
        pass

    # ---- Kill current process (strong exit)
    print("[EXIT] Shutting down process...")

    pid = os.getpid()

    # SIGTERM first (graceful), then SIGKILL fallback
    try:
        os.kill(pid, signal.SIGTERM)
    except:
        os._exit(0)

    return "🚪 App exited. You can close this tab."

# def clear_memory():
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#         gc.collect()
#     except Exception as e:
#         print("Clear memory error:", e)
# def clear_memory():
#     import gc
#     gc.collect()
#     try:
#         if torch.mps.is_available():
#             torch.mps.empty_cache()
#         elif torch.cuda.is_available():
#             torch.cuda.empty_cache()
#     except Exception as e:
#         print("Clear memory error:", e)

def clear_memory():
    import gc
    gc.collect()
    gc.collect() # Garbage collection lớp 2
    if torch.backends.mps.is_available():
        # Xóa cache cho Mac Silicon
        torch.mps.empty_cache()
        torch.mps.synchronize() # Đợi GPU hoàn tất việc xóa
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# def load_model():
#     global GENERATOR
#     if GENERATOR is None:
#         print("🚀 Loading CSM model (CPU mode)...")
#         GENERATOR = load_csm_1b("cpu")
#         print("✅ Model loaded")
#     return GENERATOR

def load_model():
    global GENERATOR
    if GENERATOR is None:
        # Tự động chọn thiết bị mạnh nhất hiện có
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
            
        print(f"🚀 Loading CSM model on {device.upper()}...")
        GENERATOR = load_csm_1b(device) 
        print(f"✅ Model loaded on {device}")
    return GENERATOR


# =========================
# 🧊 LOVE-MY-MAC MODE
# =========================
def love_my_mac_pause(enabled: bool, every_n: int, sleep_s: int, turn_idx: int):
    if not enabled:
        return

    if (turn_idx + 1) % every_n == 0:
        print(f"\n🧊 Love-my-mac pause: sleeping {sleep_s}s ...")
        clear_memory()
        time.sleep(sleep_s)


# =========================
# 🎧 BLOCK CONCAT
# =========================
def save_block(block_segments, blocks_dir: Path, block_index: int, sr: int):
    block_audio = torch.cat([s.audio for s in block_segments], dim=0)
    path = blocks_dir / f"block_{block_index:03d}.wav"
    torchaudio.save(str(path), block_audio.unsqueeze(0), sr)
    return str(path)



# =========================
# 🎙 GENERATION PIPELINE
# =========================
def run_generation(
    conversation_raw,
    speaker0,
    speaker1,
    output_dir,
    block_size,
    love_mac,
    cool_every,
    cool_seconds,
    progress=gr.Progress()
):
    global STOP_REQUESTED
    STOP_REQUESTED = False

    episode_dir = Path(output_dir)
    safe_mkdir(episode_dir)
    
    # episode_state = ensure_episode_environment(episode_dir)
    episode_state, episode_notes = ensure_episode_environment(episode_dir)
    
    session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = episode_dir / session_name

    segments_dir = session_dir / "segments"
    blocks_dir = session_dir / "blocks"

    safe_mkdir(session_dir)
    safe_mkdir(segments_dir)
    safe_mkdir(blocks_dir)


    # ---- Parse conversation
    conversation = parse_conversation(conversation_raw)
    if not conversation:
        return None, "❌ Conversation parse failed."

    # ---- Load model
    generator = load_model()

    # ---- Prepare voice prompts (ANCHORS)
    wav0 = VOICE_SAMPLE_DIR / f"{speaker0}.wav"
    wav1 = VOICE_SAMPLE_DIR / f"{speaker1}.wav"
    txt0 = VOICE_SAMPLE_DIR / f"{speaker0}.txt"
    txt1 = VOICE_SAMPLE_DIR / f"{speaker1}.txt"

    if not wav0.exists() or not wav1.exists() or not txt0.exists() or not txt1.exists():
        return None, "❌ Voice samples must include BOTH .wav and .txt"

    prompt_text_0 = txt0.read_text(encoding="utf-8").strip()
    prompt_text_1 = txt1.read_text(encoding="utf-8").strip()

    prompt_0 = prepare_prompt(prompt_text_0, 0, wav0, generator.sample_rate)
    prompt_1 = prepare_prompt(prompt_text_1, 1, wav1, generator.sample_rate)
    voice_prompts = [prompt_0, prompt_1]

    episode_state = {
        "last_session": session_name,
        "output_dir": str(episode_dir),

        "speakers": {
            "speaker_0": speaker0,
            "speaker_1": speaker1
        },

        "block_size": block_size,

        "love_my_mac": {
            "enabled": love_mac,
            "cool_every": cool_every,
            "cool_seconds": cool_seconds
        },

        "updated_at": datetime.now().isoformat()
    }

    save_episode(episode_dir, episode_state)

    # ---- Generation
    generated_segments = []
    block_segments = []
    block_files = []
    last_block_path = None
    block_index = 1

    progress(0, desc="Starting generation...")

    for i, utt in enumerate(conversation):
        if STOP_REQUESTED:
            break

        progress(i / len(conversation), desc=f"Generating turn {i+1}/{len(conversation)}")

        sliding_context = generated_segments[-2:]
        context = voice_prompts + sliding_context

        try:
            with torch.inference_mode():
                audio = generator.generate(
                    text=utt["text"],
                    speaker=utt["speaker_id"],
                    context=context,
                    max_audio_length_ms=15_000
                )

            if audio is None:
                continue

            audio = audio.cpu().float()
            seg = Segment(text=utt["text"], speaker=utt["speaker_id"], audio=audio)

            generated_segments.append(seg)
            block_segments.append(seg)

            # ---- Save per-turn
            turn_path = segments_dir / f"turn_{i+1:04d}_speaker_{utt['speaker_id']}.wav"

            torchaudio.save(str(turn_path), audio.unsqueeze(0), generator.sample_rate)

            # ---- Block save
            if len(block_segments) >= block_size:
                last_block_path = save_block(block_segments, blocks_dir, block_index, generator.sample_rate)
                block_files.append(last_block_path)
                block_segments = []
                block_index += 1
                clear_memory()

            # ---- Love-my-mac
            love_my_mac_pause(love_mac, cool_every, cool_seconds, i)

        except Exception as e:
            print("❌ Generation error:", e)

    # ---- Save remaining block
    if block_segments:
        last_block_path = save_block(block_segments, blocks_dir, block_index, generator.sample_rate)
        block_files.append(last_block_path)


    # ---- Final concat
    if generated_segments:
        final_audio = torch.cat([s.audio for s in generated_segments], dim=0)
        final_name = "final_" + session_name.replace("session_", "") + ".wav"
        final_path = session_dir / final_name

        torchaudio.save(str(final_path), final_audio.unsqueeze(0), generator.sample_rate)
    else:
        final_path = None

    clear_memory()

    status = "🛑 Stopped by user." if STOP_REQUESTED else "✅ Generation finished."

    return last_block_path, f"""{status}
    Session: {session_name}
    Blocks: {len(block_files)}
    Episode: {episode_dir}
    """
    
# =========================
# 🎨 UI / UX LAYER (GRADIO)
# =========================

def launch_gui():

    # ---- Load last session if exists
    last_output = ""
    last_block = 5
    last_love = False
    last_every = 5
    last_sleep = 10

    for folder in APP_DIR.iterdir():
        if folder.is_dir():
            data = load_session(folder)
            if data:
                last_output = data.get("output_dir", "")
                last_block = data.get("block_size", 5)
                last_love = data.get("love_my_mac", False)
                last_every = data.get("cool_every", 5)
                last_sleep = data.get("cool_seconds", 10)
    with gr.Blocks(title="Mini Studio GUI v2") as demo:
        gr.Markdown("# 🎙 Mini Studio – Voice Lab (GUI v1)")
        gr.Markdown("CSM conversation studio – strict voice anchoring (wav + txt)")

        with gr.Row():
            with gr.Column(scale=3):

                # ---------------- INPUT ----------------
                gr.Markdown("### 📝 Conversation Input")
                conversation_box = gr.Textbox(
                    lines=16,
                    placeholder="[SPEAKER_00] Hello...\n[SPEAKER_01] Hi...",
                    label="Conversation"
                )

                with gr.Row():
                    insert_btn = gr.Button("➕ Insert Order No.")
                    delete_btn = gr.Button("➖ Delete Order No.")

                # ---------------- NOTES ----------------
                gr.Markdown("### 📓 Production Notes")

                notes_box = gr.Textbox(lines=14, label="Notes (Markdown supported)")
                save_notes_btn = gr.Button("💾 Save Notes")

                # ---------------- CONTROL ----------------
                gr.Markdown("### ⚙️ Controls")

                with gr.Row():
                    generate_btn = gr.Button("🎧 Generate", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                    exit_btn = gr.Button("🚪 Exit App")

                status_box = gr.Textbox(label="Status", interactive=False)


            # =========================
            # 👉 RIGHT PANEL
            # =========================
            with gr.Column(scale=2):

                gr.Markdown("### 🎤 Voice Settings")

                voice0 = gr.Dropdown(choices=scan_voice_samples(), label="Speaker 0 voice")
                voice1 = gr.Dropdown(choices=scan_voice_samples(), label="Speaker 1 voice")

                gr.Markdown("### 📦 Block Settings")
                block_size = gr.Radio([5, 10], value=last_block, label="Turns per block")

                gr.Markdown("### 🧊 Love-my-mac Mode")
                love_toggle = gr.Checkbox(value=last_love, label="Enable love-my-mac")
                cool_every = gr.Number(value=last_every, label="Pause every N turns", precision=0)
                cool_seconds = gr.Number(value=last_sleep, label="Sleep seconds", precision=0)

                gr.Markdown("### 📂 Output Directory")

                with gr.Row():
                    output_dir = gr.Textbox(value=last_output, placeholder="/Users/.../outputs", scale=4, label=None)
                    open_folder_btn = gr.Button("📂 Open", scale=1)

                gr.Markdown("### ▶️ Latest Block Audio")
                latest_audio = gr.Audio(autoplay=True)

        # =========================
        # 🔗 EVENTS
        # =========================

        insert_btn.click(fn=insert_order_no, inputs=conversation_box, outputs=conversation_box)
        delete_btn.click(fn=delete_order_no, inputs=conversation_box, outputs=conversation_box)

        generate_btn.click(
            fn=run_generation,
            inputs=[
                conversation_box,
                voice0,
                voice1,
                output_dir,
                block_size,
                love_toggle,
                cool_every,
                cool_seconds
            ],
            outputs=[latest_audio, status_box]
        )

        stop_btn.click(fn=request_stop, outputs=status_box)

        open_folder_btn.click(fn=open_folder, inputs=output_dir, outputs=status_box)

        save_notes_btn.click(fn=save_notes, inputs=[notes_box, output_dir], outputs=status_box)

        exit_btn.click(fn=exit_app, outputs=status_box)

    demo.launch(inbrowser=True, share=False, allowed_paths=["/Volumes/SSD256"])


# =========================
# 🚀 START APP
# =========================
if __name__ == "__main__":
    launch_gui()
