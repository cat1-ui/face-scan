"""阈值定档工具（M5）。

跑完 diagnose.py 拿到候选阈值后，人工确认无误再运行本脚本，
把实测定档的阈值写入 core/config.py（替换出厂值）。

运行：.venv/Scripts/python.exe tools/set_threshold.py <阈值>
示例：.venv/Scripts/python.exe tools/set_threshold.py 0.52
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import config


def set_threshold(value):
    cfg_path = config.__file__
    with open(cfg_path, "r", encoding="utf-8") as f:
        content = f.read()

    old = f"SIMILARITY_THRESHOLD = {config.SIMILARITY_THRESHOLD}"
    new = f"SIMILARITY_THRESHOLD = {value}"
    if old not in content:
        print(f"❌ 未找到阈值常量，可能格式已变。当前值: "
              f"{config.SIMILARITY_THRESHOLD}")
        return False

    content = content.replace(old, new)
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"✅ 阈值已定档: {config.SIMILARITY_THRESHOLD} -> {value}")
    return True


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/set_threshold.py <阈值>")
        print(f"当前阈值: {config.SIMILARITY_THRESHOLD}")
        return 1
    value = sys.argv[1]
    try:
        v = float(value)
    except ValueError:
        print(f"❌ 无效阈值: {value}")
        return 1
    if not (0.0 <= v <= 1.0):
        print(f"❌ 阈值应在 0~1 之间: {value}")
        return 1
    set_threshold(v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
