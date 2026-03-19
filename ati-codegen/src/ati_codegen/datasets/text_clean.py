from __future__ import annotations

import html
import re


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def html_to_text(s: str) -> str:
    """
    LeetCode 的题面通常是 HTML；这里做最小清洗：
    - 去标签
    - HTML 实体解码
    - 规范空白与换行
    """
    if not s:
        return ""
    s = html.unescape(s)
    # 常见块级标签替换成换行，避免粘连
    s = s.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    s = s.replace("</p>", "\n").replace("</li>", "\n").replace("</pre>", "\n")
    s = _TAG_RE.sub("", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _WS_RE.sub(" ", s)
    s = _NL_RE.sub("\n\n", s)
    return s.strip()

