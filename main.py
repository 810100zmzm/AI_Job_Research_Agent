"""项目入口。

PyCharm 直接右键 Run 'main' 即可；命令行：
    python main.py                       # 不指定 --jd：跑 input/jd 下所有文件的全部岗位
    python main.py --jd input/jd/我的岗位.md --project input/project/我的项目.md
    python main.py --project input/profile/我的简历.md    # 第二个输入也可以是简历 md
    python main.py --build-resume        # 简历排版：output/resume.md + output/resume.html
    python main.py --build-resume --resume-style compact   # 换一套排版风格（classic / compact / accent）
    python main.py --help                # 全部参数（含 --llm / --vision / --answer）
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证在任意工作目录下（含 PyCharm 直接 Run）都能 import jd_agent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jd_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
