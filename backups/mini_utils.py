def parse_conversation(raw: str):
    raw = raw.strip()

    # LINE MODE ưu tiên trước
    if raw.startswith("[SPEAKER_"):
        turns = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("[SPEAKER_00]"):
                turns.append({"speaker_id": 0, "text": line.replace("[SPEAKER_00]", "").strip()})
            elif line.startswith("[SPEAKER_01]"):
                turns.append({"speaker_id": 1, "text": line.replace("[SPEAKER_01]", "").strip()})
            else:
                raise ValueError("Sai format. Mỗi dòng phải bắt đầu bằng [SPEAKER_00] hoặc [SPEAKER_01]")

        return turns

    # JSON MODE
    if raw.startswith("["):
        return json.loads(raw)

    raise ValueError("Conversation phải là JSON hoặc format [SPEAKER_xx]")
