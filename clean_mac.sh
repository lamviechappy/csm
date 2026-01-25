# File misc/clean_mac.sh (Dọn rác triệt để)

#!/bin/bash
echo "🧹 Đang dọn dẹp dự án..."

# 1. Xóa cache Python
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.py[co]" -delete

# 2. Xóa build của PyInstaller (nếu có)
rm -rf build/ dist/ *.spec

# 3. Xóa rác macOS
find . -type f -name ".DS_Store" -delete

# 4. Dọn dẹp Flutter (nếu thư mục tồn tại)
if [ -d "apps/frontend" ]; then
    cd apps/frontend && flutter clean && cd ../..
fi

echo "✨ Đã làm sạch thư mục dự án!"
