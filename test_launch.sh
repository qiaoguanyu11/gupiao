#!/bin/bash

# 设置固定的项目路径
PROJECT_PATH="/Users/qiaoguanyu/Music/gupiao"

# 确保 Python 环境正确
export PATH="/opt/homebrew/bin:$PATH"

# 激活虚拟环境
source "$PROJECT_PATH/venv/bin/activate"

# 设置PYTHONPATH并运行程序
cd "$PROJECT_PATH"
PYTHONPATH="$PROJECT_PATH" python3 "$PROJECT_PATH/src/gui_main.py" 