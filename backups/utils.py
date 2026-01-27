import re
import unicodedata

SPEAKER_PATTERN = re.compile(r"\[SPEAKER_(\d+)\]")

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa unicode để tránh lỗi tokenizer / context:
    - đổi smart quotes → '
    - bỏ ký tự control
    - chuẩn hóa form
    """

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    replacements = {
        "’": "'",
        "‘": "'",
        "“": '"',
        "”": '"',
        "…": "...",
        "–": "-",
        "—": "-",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    # bỏ ký tự control ẩn
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

    return text.strip()


def parse_conversation(raw_text: str):
    if not raw_text or not raw_text.strip():
        return []

    lines = raw_text.splitlines()

    conversation = []
    current_speaker = None
    buffer = []

    def flush():
        nonlocal buffer, current_speaker, conversation
        if current_speaker is None:
            return

        merged = " ".join(buffer).strip()
        merged = normalize_text(merged)

        if merged:
            conversation.append({
                "text": merged,
                "speaker_id": int(current_speaker)
            })

        buffer = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = SPEAKER_PATTERN.match(line)

        if match:
            flush()
            current_speaker = match.group(1)
            content = SPEAKER_PATTERN.sub("", line, count=1).strip()
            if content:
                buffer.append(content)
        else:
            buffer.append(line)

    flush()
    return conversation


import torchaudio
from generator import Segment

def load_prompt_segment(text_path, audio_path, speaker_id, target_sr):
    text = open(text_path, "r", encoding="utf-8").read().strip()

    wav, sr = torchaudio.load(audio_path)

    if wav.size(0) > 1:
        wav = wav[0]

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return Segment(
        text=text,
        speaker=speaker_id,
        audio=wav.squeeze(0).cpu()
    )
