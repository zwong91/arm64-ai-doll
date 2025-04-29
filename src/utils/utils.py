import os
import sys

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


import re
from typing import List

# 常见英文缩写（小写）——可按需补充
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    "e.g", "i.e", "etc", "vs", "st", "ft", "inc", "ltd"
}

# 正则：匹配所有可能作为终结符的标点
_TERMINATORS = r"""
    (?P<dot>\.{1,3})       |   # 英文句点 1~3 个（包括英文省略号 ...）
    (?P<zh_ellipsis>…{1,2})|   # 中文省略号 … 或 …
    (?P<end>[。！？!?])         # 中英文问号、叹号、句号
"""

_TERMINATOR_RE = re.compile(_TERMINATORS, re.UNICODE | re.VERBOSE)

def smart_split(text: str) -> List[str]:
    """
    智能分句：按中英文终结符（。！？…!?、英文省略号... 等）断句，
    跳过常见英文缩写和数字小数点，不在它们内部切分。
    """
    sentences = []
    start = 0

    for m in _TERMINATOR_RE.finditer(text):
        end = m.end()
        segment = text[start:end]

        # 如果是单个点，需要检查是否是缩写或小数点
        if m.group("dot") == ".":
            # 看前面连续字符（单词或数字）
            token = re.search(r'([A-Za-z]+|\d+)\.$', segment)
            if token:
                t = token.group(1)
                # 缩写过滤
                if t.lower() in _ABBREVIATIONS:
                    continue
                # 数字小数点过滤（如 3.14 的 .14）
                if t.isdigit():
                    continue

        sentences.append(segment.strip())
        start = end

    # 剩余部分
    if start < len(text):
        tail = text[start:].strip()
        if tail:
            sentences.append(tail)
    return sentences



