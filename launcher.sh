#!/bin/bash
# cd "$(dirname "$0")" # tự động lấy thư mục chứa
PROJECT_PATH="/Volumes/WD500/dev/csm"
cd "$PROJECT_PATH"

# Mở trình duyệt mặc định ở địa chỉ Gradio (thường là port 7860)
# Lệnh 'open' dành cho macOS, nếu dùng Linux hãy thay bằng 'xdg-open'
# open "http://127.0.0.1:7860" &

# Chạy ứng dụng
/opt/miniconda3/envs/csm_venv/bin/python "$PROJECT_PATH/mini_studio_gui_v2_1.py"
