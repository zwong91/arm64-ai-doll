import os
import sys

import re

def resource_path(path: str) -> str:
    """
    返回运行时可以访问到的绝对路径：
    1) 如果用户传入的是绝对路径，就直接返回；
    2) 否则在打包后，从 sys._MEIPASS 里找（PyInstaller onefile）；
    3) 平时开发环境，就从当前工作目录找（os.path.abspath(".")）。
    """
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(path):
        return path

    # 打包运行时，PyInstaller 会把所有资源解压到这里
    base_path = getattr(sys, "_MEIPASS", None) or os.path.abspath(".")
    return os.path.join(base_path, path)


def smart_split(text: str):
    """
    按照常见中英文终结性标点进行智能分句。
    支持中文句号、问号、叹号，英文标点和省略号等。
    """
    return re.findall(r'.*?[。！？…!?]', text, flags=re.UNICODE)

