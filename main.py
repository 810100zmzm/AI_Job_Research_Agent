"""项目入口。

PyCharm 直接右键 Run 'main' 即可；命令行：
    python main.py
    python main.py --jd input/jd/我的岗位.md --project input/project/我的项目.md
"""
from __future__ import annotations

import sys
from pathlib import Path

# 保证在任意工作目录下（含 PyCharm 直接 Run）都能 import jd_agent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jd_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
