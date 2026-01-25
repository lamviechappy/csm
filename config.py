# =========================
# csm/config.py
# =========================
import os

# Running mode: "Online" (download/check HF) or "Offline" (load from local path)
# RUNNING_MODE = "Offline"
RUNNING_MODE = "Online"


# Local model path when Offline
LOCAL_MODEL_PATH = "/Volumes/SSD256/ai-models/TTS/models--sesame--csm-1b"


# Voice samples folder
VOICE_SAMPLE_DIR = "/Volumes/SSD256/dev-projects/csm/voice_samples"


# Output folder
OUTPUT_DIR = "/Volumes/SSD256/dev-projects/csm/outputs"


# Gradio server
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7863


# Mac optimization
DEVICE = "cpu" # safest for M4 right now
DTYPE = "bfloat16"


# Love My Mac mode default
LOVE_MY_MAC_EVERY_X = 3
LOVE_MY_MAC_SLEEP_Y = 5


os.makedirs(OUTPUT_DIR, exist_ok=True)


WARMUP_PRESETS = {
    "Podcast friendly intro": """
[SPEAKER_00] Hey everyone, welcome back to the podcast. I'm really excited to be here today.
[SPEAKER_01] Me too. This is going to be a fun and relaxed conversation, so let's get started.
""",

    "Light energetic chat": """
[SPEAKER_00] Hey! It's great to see you again. How are you feeling today?
[SPEAKER_01] I'm feeling great, excited, and ready to dive into our conversation.
""",

    "None": ""
}


MAX_HISTORY_SEGMENTS = 6