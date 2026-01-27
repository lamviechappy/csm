🧱 Thứ tự lớp nên là (rất chuẩn cho project của bạn)

Trong studio_engine.py, bạn nên sắp xếp đại khái như sau:

A. imports, config
B. dataclasses (Segment, Turn, PodcastState…)
C. AudioIO   ✅
D. VoiceManager
E. WarmupManager   ✅
F. ContextBuilder
G. PodcastProject
H. CSMStudioEngine