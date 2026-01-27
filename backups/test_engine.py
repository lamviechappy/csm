from studio_engine import CSMStudioEngine

engine = CSMStudioEngine()

raw = """
[SPEAKER_0] Hello, welcome to our podcast.
[SPEAKER_1] Thanks, it's great to be here.
[SPEAKER_0] Today we talk about learning English naturally.
"""

conv = engine.parse_conversation(raw)

print(conv)
