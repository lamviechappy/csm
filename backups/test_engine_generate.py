from pathlib import Path
from studio_engine import CSMStudioEngine

engine = CSMStudioEngine()

raw = """
[SPEAKER_0] Hello everyone, welcome to our first podcast.
[SPEAKER_1] Today we’re talking about the topic "Tell me about yourself" in speaking English.
"""

conversation = engine.parse_conversation(raw)

engine.run_conversation(
    conversation=conversation,
    output_dir=Path("test_outputs"),
    file_prefix="test",
    merge_audio=True,
)
