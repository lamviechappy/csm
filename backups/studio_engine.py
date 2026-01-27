# ============================================================
# CSM STUDIO ENGINE (v2)
# File: studio_engine.py
# Part: 1/3
# Author: CSM Studio
#
# Focus:
# - Workspace & project system
# - Load/save project json
# - Voice samples loader (flat folder)
# - Parse conversation
# ============================================================

import os
import json
import re
import time
from datetime import datetime
from typing import List, Dict

from pathlib import Path
import torch
import torchaudio

import config
from generator import load_csm_1b, Segment


from typing import List

# ============================================================
# [PHẦN A] – PATH & WORKSPACE SYSTEM
# ============================================================

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[FS] Created folder: {path}")


def get_default_workspace():
    """
    Workspace dùng để chứa toàn bộ podcast projects.
    Không dùng thư mục source code để tránh bị 'đầy'.
    """
    ws = config.DEFAULT_WORKSPACE
    ensure_dir(ws)
    return ws


def get_last_project_file():
    return os.path.join(get_default_workspace(), "last_project.json")


# ============================================================
# [PHẦN B] – PROJECT SYSTEM
# ============================================================

import json
import os
from pathlib import Path
from datetime import datetime
import config

class PodcastProject:
    def __init__(self, project_name: str):
        # 1. Chuyển đổi tên project thành Path tuyệt đối
        self.project_path = Path(config.DEFAULT_WORKSPACE) / project_name
        self.project_file = self.project_path / "project.json"
        self.outputs_dir = self.project_path / "outputs"

        # 2. Tạo thư mục vật lý
        self.project_path.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

        # 3. Khởi tạo dữ liệu mặc định (Lưu ý: chưa đưa Path vào đây để tránh lỗi)
        self.data = {
            "project_name": project_name,
            "created_at": datetime.now().isoformat(),
            "audio_settings": {
                "max_audio_length_ms": config.MAX_AUDIO_LENGTH_MS,
                "temperature": config.TEMPERATURE,
                "top_k": config.TOP_K
                # "topk": config.TOP_K # Đổi thành top_k nếu lỗi, TOP_K vẫn giữ nguyên
            },
            "warmup": {
                "global_script": config.GLOBAL_WARMUP_SCRIPT,
                "regen_script": config.REGEN_WARMUP_SCRIPT
            }
        }

    def save(self):
        """Lưu dữ liệu dự án ra file JSON."""
        # Giải pháp fix lỗi PosixPath: dùng default=str
        with open(self.project_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, default=str)
        print(f"[PROJECT] Saved project.json -> {self.project_file}")

    def load(self):
        """Nạp dữ liệu từ file project.json hiện có."""
        if self.project_file.exists():
            with open(self.project_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"[PROJECT] Loaded existing config from {self.project_file}")

    def get_audio_params(self):
        """BẮT BUỘC GIỮ: Engine cần hàm này để lấy temperature, top_k..."""
        return self.data.get("audio_settings", {
            "max_audio_length_ms": config.MAX_AUDIO_LENGTH_MS,
            "temperature": config.TEMPERATURE,
            "top_k": config.TOP_K
        })

    def get_audio_output_dir(self) -> Path:
        """BẮT BUỘC GIỮ: Để biết chỗ lưu file .wav"""
        return self.outputs_dir




# ============================================================
# [PHẦN C] – VOICE SAMPLES LOADER
# ============================================================

class VoiceLibrary:
    """
    Quản lý toàn bộ voice samples trong voice_samples/
    Cấu trúc:
        voice_samples/
            anna.wav
            anna.txt
            lisa.wav
            lisa.txt
    """

    def __init__(self, voice_root: str):
        self.voice_root = voice_root
        self.voices = {}  # name -> {wav, txt}

        self.scan()

    def scan(self):
        print("\n==============================")
        print("🎤 Scanning voice samples...")
        print("==============================")

        self.voices = {}

        if not os.path.exists(self.voice_root):
            print("[VOICE] voice_samples folder not found!")
            return

        files = os.listdir(self.voice_root)
        wavs = [f for f in files if f.lower().endswith(".wav")]

        for wav in wavs:
            name = os.path.splitext(wav)[0]
            txt = name + ".txt"

            wav_path = os.path.join(self.voice_root, wav)
            txt_path = os.path.join(self.voice_root, txt)

            if not os.path.exists(txt_path):
                print(f"[VOICE] ⚠️ Missing txt for {wav}")
                continue

            self.voices[name] = {
                "wav": wav_path,
                "txt": txt_path
            }

            print(f"[VOICE] Loaded: {name}")

        print(f"[VOICE] Total voices: {len(self.voices)}")

    def list_voices(self) -> List[str]:
        return sorted(self.voices.keys())

    def get_voice(self, name: str):
        return self.voices.get(name)


# ============================================================
# [PHẦN D] – PARSE CONVERSATION
# ============================================================

def parse_conversation(raw_text: str) -> List[Dict]:
    """
    Input format:
        [SPEAKER_00] Hello...
        [SPEAKER_01] Hi...

    Output:
        [
          {"text": "...", "speaker_id": 0},
          {"text": "...", "speaker_id": 1}
        ]
    """

    print("\n[ENGINE] Parsing conversation...")
    print("--------------------------------------------------")

    conversation = []
    lines = raw_text.splitlines()

    pattern = re.compile(r"\[(SPEAKER[_ ]?0+|SPEAKER[_ ]?1+)\]\s*(.+)", re.IGNORECASE)

    for idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = pattern.match(line)
        if not match:
            print(f"[SKIP] Line {idx+1}: {line}")
            continue

        speaker_raw = match.group(1).upper()
        text = match.group(2).strip()

        if "0" in speaker_raw:
            speaker_id = 0
        elif "1" in speaker_raw:
            speaker_id = 1
        else:
            print(f"[SKIP] Unknown speaker at line {idx+1}")
            continue

        conversation.append({
            "text": text,
            "speaker_id": speaker_id
        })

        print(f"[OK] Line {idx+1} -> speaker {speaker_id}: {text[:60]}...")

    print("--------------------------------------------------")
    print(f"[ENGINE] Parsed {len(conversation)} turns\n")

    return conversation


# ============================================================
# CSM STUDIO ENGINE (v2)
# File: studio_engine.py
# Part: 2/3
#
# Focus:
# - Load CSM model
# - Voice → Segment
# - Warm-up system
# - Context builder
# ============================================================

# ============================================================
# [PHẦN E] – LOAD MODEL
# ============================================================

class CSMStudioModel:
    def __init__(self, device=None):
        print("\n==============================")
        print("🧠 Loading CSM-1B model...")
        print("==============================")

        self.device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"[MODEL] Using device: {self.device}")

        self.generator = load_csm_1b(device=self.device)
        print("[MODEL] CSM-1B loaded successfully\n")

    def get_generator(self):
        return self.generator


# ============================================================
# [PHẦN F] – VOICE → SEGMENT
# ============================================================

class VoiceSegmentBuilder:
    """
    Convert voice samples (wav+txt) into CSM Segment
    """

    def __init__(self, voice_library: VoiceLibrary, device: str):
        self.voice_library = voice_library
        self.device = device
        print(f"Các thuộc tính hiện có: {dir(self)}")

    def build_segment_old(self, text: str, speaker_id: str, generator, context: list, **kwargs):
        """
        Xây dựng một segment âm thanh từ văn bản.
        """
        print(f"   [BUILDER] Generating audio for: {text[:30]}...")
        
        with torch.inference_mode():
            # 1. Gọi generator để tạo âm thanh
            # Đảm bảo generator nhận đúng các tham số từ kwargs (max_audio_length_ms, temperature, top_k)
            audio = generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                **kwargs 
            )
            
        # 2. Chuyển audio về CPU và định dạng float32 để tránh lỗi lưu file sau này
        audio_cpu = audio.cpu().float()

        # 3. TRẢ VỀ: Phải khớp với thứ tự trong @dataclass Segment (speaker, text, audio)
        # Nếu ở generator.py bạn để @dataclass Segment: speaker, text, audio
        # Thì ở đây bạn phải truyền đúng thứ tự đó:
        return Segment(
            speaker=speaker_id, 
            text=text, 
            audio=audio_cpu
        )

    def build_segment(self, text: str, speaker_id: str, generator, context: list, **kwargs):
        print(f"   [BUILDER] Generating audio for: {text[:30]}...")
        
        # Xử lý sự khác biệt giữa 'top_k' (của project) và 'topk' (của generator)
        gen_params = kwargs.copy()
        if 'top_k' in gen_params:
            gen_params['topk'] = gen_params.pop('top_k') # Đổi top_k thành topk

        with torch.inference_mode():
            audio = generator.generate(
                text=text,
                speaker=speaker_id,
                context=context,
                **gen_params  # Truyền tham số đã được chuẩn hóa
            )
            
        audio_cpu = audio.cpu().float()

        return Segment(
            speaker=speaker_id, 
            text=text, 
            audio=audio_cpu
        )




# ============================================================
# [PHẦN I] – AUDIO IO HELPERS
# ============================================================

class AudioIO:
    TARGET_SR = 24000

    @staticmethod
    def _to_mono_1d(wav: torch.Tensor) -> torch.Tensor:
        """
        Input:
            (channels, samples) or (samples,)
        Output:
            (samples,) mono
        """
        if wav.ndim == 2:
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0)   # stereo -> mono
            else:
                wav = wav.squeeze(0)    # (1, N) -> (N,)

        return wav.contiguous().view(-1)

    @staticmethod
    def load_wav(path: Path, target_sr: int = None):
        if target_sr is None:
            target_sr = AudioIO.TARGET_SR

        wav, sr = torchaudio.load(str(path))

        wav = AudioIO._to_mono_1d(wav)

        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
            sr = target_sr

        return wav, sr

    @staticmethod
    def save_wav(wav: torch.Tensor, path: Path, sr: int = None):
        if sr is None:
            sr = AudioIO.TARGET_SR

        path.parent.mkdir(parents=True, exist_ok=True)

        wav = AudioIO._to_mono_1d(wav)

        # torchaudio expects (channels, samples)
        wav = wav.unsqueeze(0)

        torchaudio.save(str(path), wav.cpu(), sr)

    @staticmethod
    def merge_wavs(wav_files: List[Path], output_path: Path):
        merged = []
        sr = AudioIO.TARGET_SR

        for wav_path in wav_files:
            wav, _ = AudioIO.load_wav(wav_path, sr)
            merged.append(wav)

        if not merged:
            print("[MERGE] No wav files to merge.")
            return

        final = torch.cat(merged, dim=0)   # 1D concat

        AudioIO.save_wav(final, output_path, sr)








# ============================================================
# [PHẦN G] – WARM-UP SYSTEM
# ============================================================

class WarmupManager:
    # Thêm generator vào danh sách tham số truyền vào
    def __init__(self, segment_builder: VoiceSegmentBuilder, project: PodcastProject, generator):
        """
        Khởi tạo WarmupManager với đầy đủ các thành phần cần thiết.
        """
        self.segment_builder = segment_builder
        self.project = project
        self.generator = generator
        self.cached_segments: List[Segment] = []
        # THÊM DÒNG NÀY: Khởi tạo danh sách prompt trống
        self.voice_prompt_segments = [] 
        self.global_warmup_segments = []
        print(f"Các thuộc tính hiện có: {dir(self)}")
    # =====================================================
    # 🔥 BUILD GLOBAL WARM-UP SEGMENTS (VOICE PRIMING)
    # =====================================================
    # def build_global_warmup(self):
    #     print("\n[WARMUP] Building GLOBAL warm-up segments...")
    #     # Reset lại danh sách segments toàn cục mỗi khi build
    #     self.global_warmup_segments = []

    #     # Truy cập script từ dữ liệu project
    #     try:
    #         script = self.project.data["warmup"]["global_script"]
    #     except (KeyError, AttributeError) as e:
    #         print(f"[ERROR] Không tìm thấy global_script trong project data: {e}")
    #         return

    #     for i, turn in enumerate(script):
    #         print(f"[WARMUP] Global {i+1}/{len(script)} | Speaker {turn['speaker_id']}")
    #         print("   →", turn["text"])

    #         # Đảm bảo context là một danh sách hợp lệ các segments đã có trước đó
    #         # Context giúp AI duy trì sự ổn định của giọng nói
    #         current_context = self.voice_prompt_segments + self.global_warmup_segments

    #         with torch.inference_mode():
    #             # self.generator giờ đây đã được đăng ký trong __init__
    #             # Bây giờ self.generator đã tồn tại, không còn lỗi AttributeError                
    #             audio = self.generator.generate(
    #                 text=turn["text"],
    #                 speaker=turn["speaker_id"],
    #                 context=current_context,
    #                 max_audio_length_ms=config.MAX_AUDIO_LENGTH_MS,
    #                 temperature=config.TEMPERATURE,
    #                 topk=config.TOP_K,
    #             )
    #         # Chuyển dữ liệu audio về CPU để xử lý và lưu trữ
    #         audio = audio.cpu().float()

    #         # Tạo đối tượng Segment mới (Giả sử Segment là một class đã được định nghĩa)
    #         seg = Segment(text=turn["text"], speaker=turn["speaker_id"], audio=audio)
    #         self.global_warmup_segments.append(seg)

    #     print(f"[WARMUP] ✅ Global warm-up ready: {len(self.global_warmup_segments)} segments")
        
    #     # QUAN TRỌNG: Thêm dòng này để trả kết quả về cho biến bên ngoài
    #     return self.global_warmup_segments

    # def build_global_warmup(self):
    #     print("\n[WARMUP] Building GLOBAL warm-up segments...")
    #     self.global_warmup_segments = []

    #     # 1. Tạo thư mục lưu file warmup riêng để kiểm tra
    #     warmup_output_dir = self.project.project_path / "warmup_debug"
    #     warmup_output_dir.mkdir(parents=True, exist_ok=True)

    #     script = self.project.data["warmup"]["global_script"]
        
    #     # Nếu script trống, trả về list rỗng để tránh lỗi "stack expects a non-empty TensorList"
    #     if not script:
    #         print("[WARMUP] Warning: GLOBAL_WARMUP_SCRIPT is empty!")
    #         return []

    #     for i, turn in enumerate(script):
    #         print(f"[WARMUP] Global {i+1}/{len(script)} | Speaker {turn['speaker_id']}")
            
    #         # Ánh xạ tham số top_k sang topk để tránh lỗi
    #         params = self.project.get_audio_params()
    #         if 'top_k' in params:
    #             params['topk'] = params.pop('top_k')

    #         with torch.inference_mode():
    #             audio = self.generator.generate(
    #                 text=turn["text"],
    #                 speaker=turn["speaker_id"],
    #                 context=self.voice_prompt_segments + self.global_warmup_segments,
    #                 **params
    #             )
    #             # 🔧 FORCE mono 1D
    #             audio = AudioIO._to_mono_1d(audio)
    #             print("[WARMUP] audio shape after normalize:", audio.shape)
    #         audio = audio.cpu().float()
    #         seg = Segment(speaker=turn["speaker_id"], text=turn["text"], audio=audio)
    #         self.global_warmup_segments.append(seg)

    #         # 2. LƯU FILE WARMUP ĐỂ KIỂM TRA
    #         warmup_file = warmup_output_dir / f"warmup_{i+1:02d}_{turn['speaker_id']}.wav"
    #         seg.save(warmup_file)
    #         print(f"   → Debug saved: {warmup_file}")

    #     print(f"[WARMUP] ✅ Global warm-up ready: {len(self.global_warmup_segments)} segments")
    #     return self.global_warmup_segments

    def build_global_warmup(self):
        script = self.project.data["warmup"]["global_script"]
        if not script:
            print("[WARMUP] Script rỗng, bỏ qua bước tạo warmup.")
            return []

        # Thư mục debug
        debug_dir = self.project.project_path / "warmup_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        self.global_warmup_segments = []

        for i, turn in enumerate(script):
            # Kết hợp Prompts gốc + Các câu warmup đã tạo trước đó
            # Đây là mấu chốt để giữ giọng ổn định
            current_context = self.voice_prompt_segments + self.global_warmup_segments
            
            # Cắt bớt context nếu quá dài (Sliding Window sơ khai)
            if len(current_context) > 5: # Giới hạn tạm thời 5 segments để tránh nói nhảm
                current_context = current_context[-5:]

            params = self.project.get_audio_params()

            # map top_k -> topk cho đúng tên hàm generator
            if "top_k" in params:
                params["topk"] = params.pop("top_k")

            with torch.inference_mode():
                audio = self.generator.generate(
                    text=turn["text"],
                    speaker=turn["speaker_id"],
                    context=current_context,
                    **params
                )

            # ✅ CHUẨN HÓA TRƯỚC
            audio = AudioIO._to_mono_1d(audio)
            audio = audio.cpu().float()

            print("[WARMUP] normalized shape:", audio.shape)

            seg = Segment(
                speaker=turn["speaker_id"],
                text=turn["text"],
                audio=audio
            )


            self.global_warmup_segments.append(seg)

            # Debug save
            seg.save(debug_dir / f"warmup_{i:02d}_{turn['speaker_id']}.wav")

        return self.global_warmup_segments


    def get_cached_warmup(self):
        return self.cached_segments

    def clear(self):
        """Dọn dẹp bộ nhớ đệm nếu cần."""
        self.global_warmup_segments = []
        self.cached_segments = []


# ============================================================
# [PHẦN H] – CONTEXT BUILDER
# ============================================================

class ContextBuilder:
    """
    Build context for each generation call.

    Context structure (NORMAL generation):
        [voice_prompt_segments]
      + [global_warmup_segments]
      + [already_generated_segments]

    Regenerate mode sẽ có builder riêng.
    """

    def __init__(self, warmup_manager):
        self.warmup_manager = warmup_manager

    # =====================================================
    # 🔹 BASE CONTEXT = VOICE PROMPT + GLOBAL WARMUP
    # =====================================================
    def build_base_context(self) -> List[Segment]:
        base = self.warmup_manager.get_cached_warmup()

        print("\n[CONTEXT] Base context built")
        print(f"  - Total segments: {len(base)}")

        if len(base) > 0:
            print("  - First segment preview:", base[0].text[:80], "...")
            print("  - Last  segment preview:", base[-1].text[:80], "...")

        return base

    # =====================================================
    # 🔹 CONTEXT FOR NORMAL TURN GENERATION
    # =====================================================
    def build_context_for_turn(self, generated_segments: List[Segment]) -> List[Segment]:
        """
        Context used when generating a new conversation turn.

        Structure:
        voice + global warmup + generated conversation so far
        """

        base = self.build_base_context()
        full_context = base + generated_segments

        print("\n[CONTEXT] Context for new turn")
        print(f"  - Base segments      : {len(base)}")
        print(f"  - Generated segments : {len(generated_segments)}")
        print(f"  - Total context      : {len(full_context)}")

        return full_context

    # =====================================================
    # 🔹 PLACEHOLDER FOR REGENERATE (HOOK)
    # =====================================================
    def build_context_for_regenerate(
        self,
        generated_segments: List[Segment],
        target_index: int,
    ) -> List[Segment]:
        """
        Context builder for regenerate mode (HOOK ONLY, logic sẽ hoàn thiện sau).

        Dự kiến cấu trúc:
        voice + global warmup + regen warmup + last N turns
        """

        print("\n[CONTEXT] ⚠️ Regenerate context requested (not fully implemented yet)")
        print("  - target_index:", target_index)
        print("  - available segments:", len(generated_segments))

        # Tạm thời fallback về normal context
        return self.build_context_for_turn(generated_segments[:target_index])



# ============================================================
# CSM STUDIO ENGINE (v2)
# File: studio_engine.py
# Part: 3/3
#
# Focus:
# - Generate podcast
# - Save wav files
# - Merge full audio
# - Save project json
# - Love-my-Mac mode
# ============================================================





# ============================================================
# [PHẦN J] – GENERATION ENGINE
# ============================================================

class PodcastGeneratorEngine:
    def __init__(
        self,
        model: CSMStudioModel,
        project: PodcastProject,
        voice_library: VoiceLibrary,
    ):
        self.project = project
        self.model = model
        self.voice_library = voice_library

        self.generator = model.get_generator()
        self.device = model.device

        self.segment_builder = VoiceSegmentBuilder(
            voice_library=voice_library,
            device=self.device
        )

        self.warmup_manager = WarmupManager(
            segment_builder=self.segment_builder,
            project=project,
            generator=self.generator
        )

        self.context_builder = ContextBuilder(self.warmup_manager)

    # --------------------------------------------------------

    # Trong class PodcastGeneratorEngine
    def prepare_warmup(self):
        print("[Generator Engine] Building warmup...")
        if hasattr(self, 'warmup_manager'):
            # Gọi manager và trả về List các segments
            segments = self.warmup_manager.build_global_warmup()
            return segments if segments is not None else []
        return []


    # --------------------------------------------------------

    def generate_podcast(self, script_turns: List[dict]):
        print("\n==============================")
        print("🎙️ START GENERATING PODCAST")
        print("==============================")

        # 1. Lấy thư mục output từ project
        output_dir = self.project.get_audio_output_dir()

        # 2. Lấy tham số cấu hình từ project
        params = self.project.get_audio_params()

        print(f"[ENGINE] Output dir: {output_dir}")
        print(f"[ENGINE] Turns: {len(script_turns)}")

        generated_segments: List[Segment] = []
        
        # 3. Lấy context từ warmup_manager đã chuẩn bị trước đó
        # Nếu chưa chạy prepare_warmup, nó sẽ là list rỗng
        # base_context = self.warmup_manager.global_warmup_segments
        base_context = (
            self.warmup_manager.voice_prompt_segments
            + self.warmup_manager.global_warmup_segments
        )
        print("\n[CTX] Base context breakdown:")
        for idx, seg in enumerate(base_context):
            tag = "PROMPT" if seg.text == "<VOICE_PROMPT>" else "WARMUP"
            print(f"   {idx+1}. {tag} | speaker={seg.speaker} | audio_len={seg.audio.shape[-1]}")

        if "top_k" in params:
            params["topk"] = params.pop("top_k")
        for i, turn in enumerate(script_turns):
            print(f"\n[CTX] Full context size before turn {i+1}: {len(base_context + generated_segments)}")

            
            # Đảm bảo các tham số truyền vào khớp với signature ở Bước 1
            segment = self.segment_builder.build_segment(
                text=turn["text"],
                speaker_id=turn["speaker_id"],
                generator=self.generator,
                context=base_context + generated_segments,
                **params # Giải nén dict: max_audio_length_ms, temperature, top_k
            )
            
            generated_segments.append(segment)
            
            # Lưu file wav tạm thời hoặc cuối cùng
            file_path = output_dir / f"turn_{i+1:03d}_{turn['speaker_id']}.wav"
            segment.save(file_path)
            print(f"   → Saved: {file_path}")

        print(f"\n[ENGINE] ✅ Podcast generation complete!")
        return generated_segments


# ============================================================
# [PHẦN K] – HIGH LEVEL FACADE
# ============================================================

class CSMStudioEngine:
    """
    Lớp duy nhất UI cần gọi
    """

    def __init__(self, project_path: Path):
        print("\n==============================")
        print("🚀 Initializing CSM Studio Engine")
        print("==============================")

    def __init__(self, project_name: str):
        print("\n==============================")
        print(f"🚀 Initializing CSM Studio Engine: {project_name}")
        print("==============================")

        # Khởi tạo project dựa trên tên
        self.project = PodcastProject(project_name)
        
        # Nếu đã có file project.json thì nạp lại, nếu chưa thì save file mới
        if os.path.exists(self.project.project_file):
            # Hàm load() của bạn ở đây
            pass
        else:
            self.project.save()

        # self.project.load()

        self.voice_library = VoiceLibrary(config.VOICE_SAMPLE_DIR)

        self.model = CSMStudioModel()

        self.engine = PodcastGeneratorEngine(
            model=self.model,
            project=self.project,
            voice_library=self.voice_library
        )

    def list_voices(self):
        return self.voice_library.list_voices()

    def prepare_warmup(self):
        return self.engine.prepare_warmup()

    def generate(self, script: List[dict]):
        """Hàm chính để UI/Test gọi khi muốn bắt đầu tạo Podcast."""
        return self.engine.generate_podcast(script)