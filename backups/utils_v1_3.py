# utils_v1_3.py

import re
import unicodedata

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


# -----------------------
# TEXT NORMALIZATION
# -----------------------

import re
from whisper.normalizers import EnglishTextNormalizer

_whisper_norm = EnglishTextNormalizer()

def normalize_text(text: str) -> str:
    if not text:
        return ""

    # 1. whisper normalize
    text = _whisper_norm(text)

    # 2. unify quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")

    # 3. fix hyphens / dashes
    text = re.sub(r"\s*–\s*", " - ", text)
    text = re.sub(r"\s*—\s*", " - ", text)

    # tách trường hợp dính chữ: Together-where → Together - where
    text = re.sub(r"([a-zA-Z])\-([a-zA-Z])", r"\1 - \2", text)

    # 4. remove weird spacing
    text = re.sub(r"\s+", " ", text)

    # 5. fix space before punctuation
    text = re.sub(r"\s+([?.!,])", r"\1", text)

    return text.strip()



# -----------------------
# CONVERSATION PARSER
# -----------------------

def parse_conversation(text: str):
    """
    Input:
        [SPEAKER_00] Hello...
        [SPEAKER_01] Hi...

    Output:
        [
          {"text": "...", "speaker_id": 0},
          {"text": "...", "speaker_id": 1}
        ]
    """
    if not text:
        return []

    pattern = r"\[SPEAKER_(\d+)\]\s*(.*?)(?=\s*\[SPEAKER_|\Z)"
    matches = re.findall(pattern, text, re.DOTALL)

    conversation = []
    for s, c in matches:
        cleaned = c.strip().replace("\n", " ")
        cleaned = normalize_text(cleaned)

        if cleaned:
            conversation.append({
                "text": cleaned,
                "speaker_id": int(s)
            })

    return conversation
