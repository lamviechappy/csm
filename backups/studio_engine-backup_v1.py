"""
============================================================
CSM STUDIO - CORE ENGINE
File: studio_engine.py

Chức năng:
- Load CSM model
- Load voice samples
- Build warm-up context
- Parse conversation
- Generate full podcast audio
- Hook regenerate

Phần 1:
- imports
- init engine
- load model
- load voice samples
- warm-up builder
============================================================
"""

import os
import json
from pathlib import Path
from typing import List, Dict

import torch
import torchaudio

from generator import load_csm_1b, Segment

import config


# ============================================================
# [PHẦN A] – DEVICE & MODEL LOADING
# ============================================================

def detect_device():
    if torch.backends.mps.is_available():
        print("[ENGINE] Using MPS")
        return "mps"
    elif torch.cuda.is_available():
        print("[ENGINE] Using CUDA")
        return "cuda"
    else:
        print("[ENGINE] Using CPU")
        return "cpu"


class CSMStudioEngine:

    def __init__(self):
        print("\n==============================")
        print("🚀 Initializing CSM Studio Engine")
        print("==============================")

        self.device = detect_device()
        self.generator = None
        self.sample_rate = None

        self.voice_prompt_segments: List[Segment] = []
        self.global_warmup_segments: List[Segment] = []

        self.load_model()
        self.load_voice_samples()
        self.build_global_warmup()

        print("✅ Engine ready.\n")


    # ========================================================
    # [PHẦN B] – LOAD MODEL
    # ========================================================

    def load_model(self):
        print("[ENGINE] Loading CSM model...")
        self.generator = load_csm_1b(device=self.device)
        self.sample_rate = self.generator.sample_rate
        print("[ENGINE] Model loaded. Sample rate:", self.sample_rate)


    # ========================================================
    # [PHẦN C] – AUDIO LOADER
    # ========================================================

    def load_audio(self, audio_path: Path) -> torch.Tensor:
        print(f"[ENGINE] Loading audio: {audio_path}")
        wav, sr = torchaudio.load(audio_path)
        wav = wav.squeeze(0)

        if sr != self.sample_rate:
            print(f"[ENGINE] Resampling {sr} -> {self.sample_rate}")
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        return wav


    # ========================================================
    # [PHẦN D] – LOAD VOICE SAMPLES (VOICE PROMPTS)
    # ========================================================

    def load_voice_samples(self):
        """
        Load voice samples từ voice_samples/speaker_x/*.wav
        Dùng làm base voice identity
        """

        print("[ENGINE] Loading voice samples...")

        self.voice_prompt_segments = []

        for speaker_id, folder_name in config.VOICE_SAMPLE_STRUCTURE.items():
            speaker_folder = config.VOICE_SAMPLE_DIR / folder_name

            if not speaker_folder.exists():
                print(f"⚠️ Voice folder not found: {speaker_folder}")
                continue

            wav_files = list(speaker_folder.glob("*.wav"))
            print(f"[ENGINE] Speaker {speaker_id}: {len(wav_files)} samples")

            for wav_path in wav_files:
                audio = self.load_audio(wav_path)
                seg = Segment(
                    text="[VOICE PROMPT]",
                    speaker=speaker_id,
                    audio=audio
                )
                self.voice_prompt_segments.append(seg)

        print(f"[ENGINE] Total voice prompt segments: {len(self.voice_prompt_segments)}")


    # ========================================================
    # [PHẦN E] – BUILD GLOBAL WARM-UP CONTEXT
    # ========================================================

    def build_global_warmup(self):
        """
        Sinh audio warm-up hard-code để tạo voiceover context
        """

        print("[ENGINE] Building global warm-up context...")

        self.global_warmup_segments = []

        base_context = list(self.voice_prompt_segments)

        for i, turn in enumerate(config.GLOBAL_WARMUP_SCRIPT):
            print(f"[WARMUP] Generating {i+1}/{len(config.GLOBAL_WARMUP_SCRIPT)}")

            with torch.inference_mode():
                audio = self.generator.generate(
                    text=turn["text"],
                    speaker=turn["speaker_id"],
                    context=base_context + self.global_warmup_segments,
                    max_audio_length_ms=config.MAX_AUDIO_LENGTH_MS,
                )

            if audio is None:
                print("⚠️ Warm-up returned None")
                continue

            audio = audio.cpu().float()

            seg = Segment(
                text=turn["text"],
                speaker=turn["speaker_id"],
                audio=audio
            )

            self.global_warmup_segments.append(seg)

        print("[ENGINE] Global warm-up ready:", len(self.global_warmup_segments), "segments")


# ============================================================
# [PHẦN F] – PARSE CONVERSATION
# ============================================================

    def parse_conversation(self, raw_text: str) -> List[Dict]:
        """
        Input format:
        [SPEAKER_00] Hello...
        [SPEAKER_01] Hi...

        Output:
        [
            {"text": "...", "speaker_id": 0},
            {"text": "...", "speaker_id": 1},
        ]
        """

        print("\n[ENGINE] Parsing conversation...")
        print("--------------------------------------------------")

        conversation = []

        lines = raw_text.splitlines()

        for idx, line in enumerate(lines):
            line = line.strip()

            if not line:
                continue

            if not line.startswith("[") or "]" not in line:
                print(f"⚠️ Skipped line {idx+1}: {line}")
                continue

            try:
                header, content = line.split("]", 1)
                speaker_raw = header.replace("[", "").replace("]", "").strip()

                if not speaker_raw.upper().startswith("SPEAKER"):
                    print(f"⚠️ Invalid speaker tag at line {idx+1}: {speaker_raw}")
                    continue

                speaker_id = int(speaker_raw.split("_")[-1])

                text = content.strip()
                if not text:
                    print(f"⚠️ Empty content at line {idx+1}")
                    continue

                conversation.append({
                    "text": text,
                    "speaker_id": speaker_id
                })

                print(f"[OK] Line {idx+1} -> speaker {speaker_id}: {text[:60]}...")

            except Exception as e:
                print(f"❌ Parse error line {idx+1}: {line}")
                print("   ", str(e))

        print("--------------------------------------------------")
        print(f"[ENGINE] Parsed {len(conversation)} turns\n")

        return conversation


# ============================================================
# [PHẦN G] – BUILD REGENERATE WARM-UP
# ============================================================

    def build_regenerate_context(
        self,
        generated_segments: List[Segment],
        regen_index: int,
        warmup_turns: int,
    ) -> List[Segment]:
        """
        Tạo context khi regenerate câu thứ N
        - lấy x câu trước đó
        - + hard-code regen warmup script
        """

        print("\n[ENGINE] Building regenerate context...")
        print(f"[ENGINE] Regen index: {regen_index}")
        print(f"[ENGINE] Warmup turns: {warmup_turns}")

        start = max(0, regen_index - warmup_turns)
        base_segments = generated_segments[start:regen_index]

        print(f"[ENGINE] Using {len(base_segments)} previous segments")

        regen_segments = []

        for i, turn in enumerate(config.REGENERATE_WARMUP_SCRIPT):
            print(f"[REGEN WARMUP] {i+1}/{len(config.REGENERATE_WARMUP_SCRIPT)}")

            with torch.inference_mode():
                audio = self.generator.generate(
                    text=turn["text"],
                    speaker=turn["speaker_id"],
                    context=self.voice_prompt_segments + base_segments + regen_segments,
                    max_audio_length_ms=config.MAX_AUDIO_LENGTH_MS,
                )

            if audio is None:
                print("⚠️ Regen warm-up returned None")
                continue

            audio = audio.cpu().float()

            seg = Segment(
                text=turn["text"],
                speaker=turn["speaker_id"],
                audio=audio
            )

            regen_segments.append(seg)

        print("[ENGINE] Regen warm-up ready:", len(regen_segments), "segments\n")

        return self.voice_prompt_segments + base_segments + regen_segments


# ============================================================
# [PHẦN H] – SAVE AUDIO
# ============================================================

    def save_audio(self, audio: torch.Tensor, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(out_path.as_posix(), audio.unsqueeze(0), self.sample_rate)
        print(f"[ENGINE] Saved: {out_path.name}")


# ============================================================
# [PHẦN I] – GENERATE FULL CONVERSATION
# ============================================================

    def run_conversation(
        self,
        conversation: List[Dict],
        output_dir: Path,
        file_prefix: str = "pod",
        merge_audio: bool = True,
        love_my_mac: bool = True,
    ):
        print("\n==============================")
        print("🎙️ START GENERATING PODCAST")
        print("==============================")
        print(f"[ENGINE] Output dir: {output_dir}")
        print(f"[ENGINE] Turns: {len(conversation)}")

        output_dir.mkdir(parents=True, exist_ok=True)

        generated_segments: List[Segment] = []
        saved_files: List[Path] = []

        base_context = self.voice_prompt_segments + self.global_warmup_segments

        for i, utterance in enumerate(conversation):
            print(f"\n[ENGINE] Generating ({i+1}/{len(conversation)})")
            print("Speaker:", utterance["speaker_id"])
            print("Text:", utterance["text"][:120])

            try:
                with torch.inference_mode():
                    audio_tensor = self.generator.generate(
                        text=utterance["text"],
                        speaker=utterance["speaker_id"],
                        context=base_context + generated_segments,
                        max_audio_length_ms=config.MAX_AUDIO_LENGTH_MS,
                    )

                if audio_tensor is None:
                    print("⚠️ Returned None")
                    continue

                audio_tensor = audio_tensor.cpu().float()

                seg = Segment(
                    text=utterance["text"],
                    speaker=utterance["speaker_id"],
                    audio=audio_tensor
                )
                generated_segments.append(seg)

                filename = f"{file_prefix}_{i+1:03d}.wav"
                out_path = output_dir / filename
                self.save_audio(audio_tensor, out_path)
                saved_files.append(out_path)

                # # Love-my-Mac mode
                # if love_my_mac and (i + 1) % config.LOVE_MY_MAC_EVERY == 0:
                #     print(f"🍏 Love-my-Mac pause {config.LOVE_MY_MAC_SLEEP}s")
                #     torch.mps.empty_cache()
                #     import time
                #     time.sleep(config.LOVE_MY_MAC_SLEEP)

                # Love-my-Mac mode (safe)
                try:
                    if config.LOVE_MY_MAC and (i + 1) % config.LOVE_MY_MAC_EVERY == 0:
                        print(f"🍏 Love-my-Mac pause {config.LOVE_MY_MAC_SLEEP}s")
                        torch.mps.empty_cache()
                        import time
                        time.sleep(config.LOVE_MY_MAC_SLEEP)
                except Exception as e:
                    print("⚠️ [WARN] Love-my-Mac config error:", e)
                
            except Exception as e:
                print(f"❌ Error at sentence {i+1}: {str(e)}")
                break

        # ===========================
        # MERGE AUDIO
        # ===========================

        merged_path = None
        if merge_audio and saved_files:
            print("\n[ENGINE] Merging audio files...")
            merged = torch.cat(
                [torchaudio.load(p)[0].squeeze(0) for p in saved_files]
            )
            merged_path = output_dir / f"{file_prefix}_FULL.wav"
            torchaudio.save(merged_path.as_posix(), merged.unsqueeze(0), self.sample_rate)
            print("[ENGINE] Merged file:", merged_path)

        print("\n✅ GENERATION COMPLETE")
        return saved_files, merged_path


# ============================================================
# [PHẦN J] – REGENERATE ONE SENTENCE
# ============================================================

    def regenerate_sentence(
        self,
        conversation: List[Dict],
        output_dir: Path,
        sentence_index: int,
        warmup_turns: int = None,
        file_prefix: str = "pod",
    ):
        print("\n==============================")
        print("🔁 REGENERATING SENTENCE")
        print("==============================")

        if warmup_turns is None:
            warmup_turns = config.REGENERATE_WARMUP_TURNS

        print("[ENGINE] Target index:", sentence_index)
        print("[ENGINE] Warmup turns:", warmup_turns)

        # Load generated segments before this sentence
        generated_segments = []

        for i in range(sentence_index):
            wav_path = output_dir / f"{file_prefix}_{i+1:03d}.wav"
            if not wav_path.exists():
                continue

            audio = self.load_audio(wav_path)
            seg = Segment(
                text=conversation[i]["text"],
                speaker=conversation[i]["speaker_id"],
                audio=audio
            )
            generated_segments.append(seg)

        regen_context = self.build_regenerate_context(
            generated_segments=generated_segments,
            regen_index=sentence_index,
            warmup_turns=warmup_turns,
        )

        target = conversation[sentence_index]

        print("\n[ENGINE] Regenerating:")
        print(target["text"])

        with torch.inference_mode():
            audio = self.generator.generate(
                text=target["text"],
                speaker=target["speaker_id"],
                context=regen_context,
                max_audio_length_ms=config.MAX_AUDIO_LENGTH_MS,
            )

        if audio is None:
            print("❌ Regen failed (None)")
            return None

        audio = audio.cpu().float()

        out_path = output_dir / f"{file_prefix}_{sentence_index+1:03d}.wav"
        self.save_audio(audio, out_path)

        print("✅ Regen complete:", out_path)
        return out_path
