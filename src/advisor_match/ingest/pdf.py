"""论文 PDF 解析。

产出 ``ParsedDocument``：纯文本 + 页码定位，供台账抽取出 ``SourceRef``。
注意：解析失败时返回 None 并记录原因，不做 OCR 猜测。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    doc_id: str
    path: Path
    pages: list[ParsedPage]

    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)


def parse_pdf(path: Path, *, doc_id: str) -> ParsedDocument | None:
    """用 PyMuPDF 抽取文本。

    TODO: 处理双栏排版与页眉页脚；扫描版 PDF 先用 OCR 转文本再做，不要假装解析成功。
    """
    raise NotImplementedError("由韩宇轩实现：PyMuPDF 文本抽取 + 双栏重排")
