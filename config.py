"""
============================================================
CSM STUDIO - CONFIG FILE
Author: (your project)
Purpose:
- Central place for all constants & configuration
- Easy to tweak without touching main logic
============================================================

Python version target: 3.11
"""

import os
from pathlib import Path


# ============================================================
# [PHẦN ROOT] – AUTO DETECT PROJECT ROOT
# ============================================================

"""
Mục tiêu:
- Xác định thư mục gốc của project một cách tự động
- Không phụ thuộc vào nơi chạy file
- An toàn khi build app / đổi folder

Logic:
- Lấy đường dẫn của chính file config.py
- Đi ngược lên 1 cấp làm project root
"""

CONFIG_FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = CONFIG_FILE_PATH.parent   # nếu config.py nằm ở root project
# Nếu sau này bạn chuyển config.py vào thư mục /core
# thì đổi thành:
# PROJECT_ROOT = CONFIG_FILE_PATH.parent.parent




# ============================================================
# [PHẦN A] – APP INFO
# ============================================================

APP_NAME = "CSM Studio"
APP_VERSION = "0.1.0"
DEBUG = True   # bật nhiều print debug


# ============================================================
#  TẠO WORKSPACE FOLDER
# ============================================================

def get_workspace_path():
    """Trả về đối tượng Path dẫn đến thư mục CSM_Projects trong thư mục Home."""
    return Path.home() / "CSM_Projects"

def setup_default_workspace():
    """Tạo thư mục nếu chưa có và thiết lập làm thư mục làm việc chính."""
    workspace = get_workspace_path()
    
    # Tạo thư mục (parents=True giúp tạo cả thư mục cha nếu cần, exist_ok=True tránh lỗi nếu đã tồn tại)
    workspace.mkdir(parents=True, exist_ok=True)
    
    # Thay đổi thư mục làm việc của hệ thống (Current Working Directory)
    os.chdir(workspace)
    
    return str(workspace)

# Bạn có thể khai báo thêm các biến cấu hình khác dựa trên workspace này
DEFAULT_WORKSPACE = get_workspace_path()
LOGS_DIR = DEFAULT_WORKSPACE / "logs"


print(f"[CONFIG] CONFIG_FILE_PATH = {CONFIG_FILE_PATH}")
print(f"[CONFIG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[CONFIG] DEFAULT_WORKSPACE = {DEFAULT_WORKSPACE}")



# ============================================================
# [PHẦN B] – PROJECT STRUCTURE
# ============================================================

# File lưu cấu hình project (conversation, output folder, regenerate history…)
PROJECT_CONFIG_NAME = "project.json"

# Thư mục con mặc định trong project

DEFAULT_OUTPUT_DIRNAME = "outputs"
DEFAULT_TEMP_DIRNAME = "temp"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / DEFAULT_OUTPUT_DIRNAME
DEFAULT_TEMP_DIR = PROJECT_ROOT / DEFAULT_TEMP_DIRNAME

# Các format file audio xuất ra
AUDIO_EXT = ".wav"
 
# ============================================================
# [PHẦN C] – CSM / MODEL CONFIG
# ============================================================

# Tên model huggingface
CSM_MODEL_NAME = "sesame/csm-1b"

# In ra nhiều log khi generate
PRINT_EACH_STEP = True


# ============================================================
# [PHẦN D] – SPEAKER CONFIG (GIAI ĐOẠN 1: CỐ ĐỊNH 2 SPEAKER)
# ============================================================

# Giai đoạn này chỉ dùng 2 speaker
# FIXED_SPEAKERS = [0, 1]

# Label để hiện trên UI
# SPEAKER_LABELS = {
#     0: "Speaker A",
#     1: "Speaker B",
# }

# Thư mục chứa voice samples (giọng mẫu)
# VOICE_SAMPLE_DIR = "voice_samples"
VOICE_SAMPLE_DIR = PROJECT_ROOT / "voice_samples"


# Mapping: speaker_id -> list file wav
# (sẽ load động trong app, đây chỉ là tên thư mục gốc)
# VOICE_SAMPLE_STRUCTURE = {
#     0: "speaker_0",
#     1: "speaker_1",
# }


# ============================================================
# [PHẦN E] – HARD-CODE WARM-UP (VOICE OVER CONTEXT)
# ============================================================

"""
Warm-up này KHÔNG phụ thuộc conversation.
Mục đích:
- Tạo voice base ổn định
- Định hình style podcast

Sau này có thể:
- sinh warm-up bằng LLM
- hoặc cho user nhập
"""

# GLOBAL_WARMUP_SCRIPT = [
#     {"text": "Welcome back to our podcast. I’m really happy to be here with you today.", "speaker_id": 0},
#     {"text": "Me too. This is one of my favorite moments of the day, just talking and sharing stories.", "speaker_id": 1},
#     {"text": "Let’s take a deep breath, relax, and enjoy this conversation together.", "speaker_id": 0},
# ]

# GLOBAL_WARMUP_SCRIPT = [
#     {"text": "Welcome back to our podcast. I’m really happy to be here with you today.", "speaker_id": 0},
#     {"text": "Me too. This is one of my favorite moments of the day, just talking and sharing stories.", "speaker_id": 1},
# ]

GLOBAL_WARMUP_SCRIPT = [
    {"text": "Welcome back to our podcast.", "speaker_id": 0},
    {"text": "This is one of my favorite moments of the day, just talking and sharing stories.", "speaker_id": 1},
]

# GLOBAL_WARMUP_SCRIPT =[]

"""
Warm-up cho regenerate (khi làm lại câu hỏng).
Dùng khi user regenerate turn N, hệ thống sẽ:
- lấy X câu trước đó
- cộng thêm đoạn warm-up này
"""

REGEN_WARMUP_SCRIPT = [
    {"text": "Before we continue, let’s get back into the flow of our conversation.", "speaker_id": 0},
    {"text": "Yes, let’s keep the same energy and tone as before.", "speaker_id": 1},
]

REGEN_WARMUP_TURNS = 3


# ============================================================
# [PHẦN F] – PARSE CONVERSATION RULES
# ============================================================

"""
Format conversation input:

[SPEAKER_00] Hello...
[SPEAKER_01] Hi...
[SPEAKER_00] How are you?

Hoặc:
[SPEAKER00]
[SPEAKER_0]
[SPEAKER_1]
"""

SPEAKER_PREFIXES = [
    "SPEAKER_",
    "SPEAKER",
]

ALLOW_EMPTY_LINE = True


# ============================================================
# [PHẦN G] – REGENERATE CONFIG
# ============================================================

# Mặc định khi regenerate:
# lấy bao nhiêu câu trước đó làm context
DEFAULT_REGEN_PREV_TURNS = 3

# Lưu file regenerate log
REGEN_LOG_NAME = "regenerate_log.json"


# ============================================================
# [PHẦN H] – UI DEFAULTS
# ============================================================

DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 800

DEFAULT_CONVERSATION_PLACEHOLDER = """[SPEAKER_0] Hello everyone, welcome back to our podcast.
[SPEAKER_1] We’re really happy you’re here with us today.
[SPEAKER_0] Today’s topic is something many people struggle with: introducing yourself naturally.
"""


# ============================================================
# [PHẦN I] – INTERNAL UTILS
# ============================================================

# def ensure_dir(path: str | Path):
#     """Tạo folder nếu chưa tồn tại"""
#     path = Path(path)
#     if not path.exists():
#         if DEBUG:
#             print(f"[CONFIG] Creating directory: {path}")
#         path.mkdir(parents=True, exist_ok=True)
#     return path


# chắc chắn dùng Path
def ensure_dir(path: str | Path):
    path = Path(path).resolve()
    if not path.exists():
        if DEBUG:
            print(f"[CONFIG] Creating directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    return path


def debug_print(msg: str):
    if DEBUG:
        print(msg)


# ==============================
# 🍏 LOVE MY MAC MODE (anti-freeze)
# ==============================

LOVE_MY_MAC = True               # bật / tắt
LOVE_MY_MAC_EVERY = 5            # cứ bao nhiêu câu thì nghỉ
LOVE_MY_MAC_SLEEP = 60            # nghỉ bao nhiêu giây


# ==============================
# 🎛️ AUDIO GENERATION SETTINGS
# ==============================

# Giới hạn độ dài mỗi câu (ms)
MAX_AUDIO_LENGTH_MS = 10_000

# Khi warm-up, chỉ lấy tối đa N segment cuối (tránh context quá dài)
MAX_CONTEXT_SEGMENTS = 12

# Sampling control
TEMPERATURE = 0.8        # 0.3–1.0 (thấp = ổn định, cao = sáng tạo)
TOP_K = 50               # giới hạn sampling
# TOP_P = 0.9              # nucleus sampling (nếu model hỗ trợ)

# Repetition / stability (để chừa sẵn)
REPETITION_PENALTY = 1.0

# Debug
PRINT_GENERATION_PARAMS = True


# ============================================================
# [PHẦN J] – QUICK SELF TEST
# ============================================================

if __name__ == "__main__":
    print("===== CSM STUDIO CONFIG TEST =====")
    print("APP:", APP_NAME, APP_VERSION)
    print("Model:", CSM_MODEL_NAME)
    print("Fixed speakers:", FIXED_SPEAKERS)
    print("Output dirname default:", DEFAULT_OUTPUT_DIRNAME)
    print("Output dir default:", DEFAULT_OUTPUT_DIR)
    print("Temp dirname default:", DEFAULT_TEMP_DIRNAME)
    print("Temp dir default:", DEFAULT_TEMP_DIR)
    print("Global warmup turns:", len(GLOBAL_WARMUP_SCRIPT))
    print("Regenerate warmup turns:", len(REGEN_WARMUP_SCRIPT))

