"""材料文本解析：把上传的 txt/md 转成台账候选陈述。

只做本地文本处理，不联网、不保存文件。每条陈述会成为一条台账记录，
因此长度限制与台账一致（不超过 300 字）；超长的行直接跳过并在界面提示，
而不是悄悄截断成一句不完整的话。
"""

from __future__ import annotations

import hashlib
import io
import re

MAX_LINES = 50
MAX_CHARS = 300

_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.、)]|\(\d+\))\s*")
_HEADING = re.compile(r"^\s*#+\s*")
_CN_SECTION = re.compile(r"^[一二三四五六七八九十]+[、.]\s*")
_CONTACT = re.compile(r"1[3-9]\d{9}|[\w.+-]+@[\w.-]+|\d{17}[\dXx]")

#: 带标签的联系方式，中英文都常见：``Phone: ...``、``邮箱：...``。
#: 从标签删到行尾，避免把号码、邮箱或后面的文字留在台账里。
_CONTACT_LABEL = re.compile(
    r"(?:电子邮箱|手机号码|联系电话|联系方式|电话|手机|邮箱|地址|"
    r"e-?mail|phone|mobile|tel|telephone|wechat|weixin|linkedin|github|website|homepage)"
    r"\s*[:：].*$",
    re.I,
)
_NAME_LABEL = re.compile(r"^(?:姓名|name)\s*[:：]", re.I)

#: 英文姓名：2–3 个首字母大写的单词（"Alex Chen" / "Alex J. Chen"）。
_LATIN_NAME = re.compile(r"^[A-Z][A-Za-z.'-]*(?:\s+[A-Z][A-Za-z.'-]*){1,2}$")
#: 这些词开头的两词行更像标题/机构/奖项，不应当成姓名。
_LATIN_NAME_STOP = {
    "basic", "personal", "contact", "research", "projects", "skills", "education",
    "awards", "honors", "summary", "university", "college", "institute", "school",
    "department", "national", "scholarship", "computer", "vision", "science",
    "engineering", "information", "machine", "learning", "data", "artificial",
    "intelligence",
}

_DEGREE = re.compile(r"\b(?:B\.?Eng|B\.?Sc?|M\.?S|M\.?Eng|Ph\.?D|Bachelor|Master|Doctor)\b", re.I)
_CN_EDU = re.compile(r"^(?:本科|硕士|博士|研究生|教育背景|教育经历)")

#: 简历里常见的章节名。它们不是成果，解析时必须丢掉。
_SECTION_TITLES = {
    "基本信息", "个人信息", "联系方式", "个人资料", "教育背景", "教育经历", "学习经历",
    "工作经历", "实习经历", "项目经历", "项目经验", "科研经历", "研究经历", "研究兴趣",
    "研究方向", "校园经历", "社会实践", "专业技能", "技能", "技能特长", "获奖情况",
    "获奖经历", "荣誉奖项", "奖励荣誉", "论文发表", "发表论文", "学术成果", "研究成果",
    "科研项目", "自我评价", "个人评价", "兴趣爱好", "语言能力", "其他", "附录",
    "科研兴趣", "学术兴趣", "研究领域", "研究专长", "主修课程", "核心课程", "课程成绩",
    "获奖", "证书", "技能证书", "校园活动", "个人简历", "简历", "论文", "专利",
    "basicinformation", "personalinformation", "contact", "education",
    "educationbackground", "experience", "workexperience", "internship",
    "projects", "projectexperience", "skills", "publications", "awards",
    "honors", "summary", "research", "researchinterests", "researchexperience",
    "researchprojects", "coursework", "certificates", "languages",
    "extracurricular", "activities", "objective", "profile",
}


#: 常见姓氏，用于识别"名字"这种不该进入台账的短行。
_SURNAMES = set(
    "王李张刘陈杨黄赵吴周徐孙马朱胡郭何高林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于"
    "蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范金石廖贾夏韦付方白邹孟熊秦邱江尹薛"
    "闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤"
)


def _looks_like_name(line: str) -> bool:
    text = line.replace(" ", "").replace("　", "")
    if not 2 <= len(text) <= 4:
        return False
    if not all("一" <= char <= "鿿" for char in text):
        return False
    return text[0] in _SURNAMES


def _looks_like_latin_name(line: str) -> bool:
    """识别 "Alex Chen" 这类英文姓名行；标题、机构、奖项不算。"""
    text = " ".join(line.split())
    text = _CONTACT_LABEL.sub("", text).strip(" ,;|")
    if not 2 <= len(text.split()) <= 3 or len(text) > 40:
        return False
    if not _LATIN_NAME.match(text):
        return False
    return not any(word.lower() in _LATIN_NAME_STOP for word in text.split())


def _is_education_line(line: str) -> bool:
    """教育背景不是成果；按行解析时直接跳过，避免混进台账。"""
    text = line.strip()
    if re.match(r"(?i)^(?:GPA|Education)\b", text):
        return True
    if _CN_EDU.match(text) and ("大学" in text or "学院" in text):
        return True
    return bool(
        _DEGREE.search(text)
        and any(key in text for key in ("University", "College", "Institute"))
    )


def _normalize_heading(line: str) -> str:
    return (
        line.strip()
        .strip("：:")
        .replace(" ", "")
        .replace("　", "")
        .lower()
    )


def _is_section_title(line: str) -> bool:
    normalized = _normalize_heading(line)
    if normalized in _SECTION_TITLES:
        return True
    # 允许"教育背景 Education""项目经历（续）"这类带后缀的标题；
    # "技能：Python" 这种带内容的行不是标题。
    for title in _SECTION_TITLES:
        if not normalized.startswith(title):
            continue
        rest = normalized[len(title) :]
        if len(rest) > 12 or "：" in rest or ":" in rest:
            continue
        return True
    return False


def statements_from_text(
    text: str, *, max_lines: int = MAX_LINES, max_chars: int = MAX_CHARS
) -> tuple[list[str], list[str]]:
    """返回（陈述列表, 提示信息）。

    处理：去项目符号/序号、跳过空行与 Markdown 标题、去重、超长行跳过、总量截断。
    """
    warnings: list[str] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    statements: list[str] = []
    seen: set[str] = set()
    skipped_long = 0
    duplicates = 0
    removed_contacts = 0
    skipped_education = 0

    for line_no, raw in enumerate(lines):
        if _HEADING.match(raw):
            continue
        line = _BULLET.sub("", _CN_SECTION.sub("", raw)).strip()
        if _CONTACT_LABEL.search(line):
            line = _CONTACT_LABEL.sub("", line).strip(" ：:，,;|")
            removed_contacts += 1
        if not line:
            continue
        if _is_section_title(line) or _looks_like_name(line):
            continue
        if line_no < 10 and _looks_like_latin_name(line):
            continue
        if _CONTACT.search(line):
            line = _CONTACT.sub("", line).strip(" ：:，,;|")
            removed_contacts += 1
            if len(line) < 4:
                continue
        if _is_education_line(line):
            skipped_education += 1
            continue
        if len(line) > max_chars:
            skipped_long += 1
            continue
        if line in seen:
            duplicates += 1
            continue
        seen.add(line)
        statements.append(line)

    if skipped_long:
        warnings.append(f"{skipped_long} 行超过 {max_chars} 字，已跳过")
    if duplicates:
        warnings.append(f"{duplicates} 行重复，已去重")
    if removed_contacts:
        warnings.append(f"{removed_contacts} 行包含联系方式，已移除")
    if skipped_education:
        warnings.append(f"{skipped_education} 行教育背景，已跳过")
    if len(statements) > max_lines:
        warnings.append(f"超过 {max_lines} 条，只保留前 {max_lines} 条")
        statements = statements[:max_lines]

    return statements, warnings


def _block_chunks(raw: str) -> list[str]:
    """按块内换行拆开：标题、姓名、联系方式单独成行，段落内的折行仍然合并。"""
    chunks: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            chunks.append(" ".join(buffer))
            buffer.clear()

    for part in raw.splitlines():
        line = " ".join(part.split())
        if not line:
            continue
        if (
            _CONTACT_LABEL.search(line)
            or _is_section_title(line)
            or _looks_like_name(line)
            or _looks_like_latin_name(line)
        ):
            flush()
            chunks.append(line)
        else:
            buffer.append(line)
    flush()
    return chunks


def extract_pdf_text(data: bytes, *, max_pages: int = 20) -> tuple[str, list[str]]:
    """抽取 PDF 文本，按文本块合并行，避免把一句话拆成多条台账。

    扫描版 PDF 没有文本层，这里不做 OCR，只如实提示。
    """
    try:
        import fitz
    except ImportError:
        return "", ["未安装 PyMuPDF，无法解析 PDF"]

    warnings: list[str] = []
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as error:  # noqa: BLE001 - 损坏文件不应让界面崩溃
        return "", [f"PDF 打开失败：{type(error).__name__}"]

    if document.needs_pass:
        return "", ["PDF 有密码保护，无法解析"]

    page_limit = min(document.page_count, max_pages)
    chunks: list[str] = []
    for index in range(page_limit):
        for block in document[index].get_text("blocks"):
            chunks.extend(_block_chunks(str(block[4])))

    if document.page_count > max_pages:
        warnings.append(f"仅解析前 {max_pages} 页")
    if not chunks:
        warnings.append("未提取到文字，可能是扫描版 PDF（未做 OCR）")
    document.close()
    return "\n".join(chunks), warnings


def extract_signature(text: str, *, max_affiliation: int = 30) -> str | None:
    """从材料里提取署名：姓名 + 学校/院系行（如果有）。"""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    name: str | None = None

    match = re.search(
        r"(?:姓名|Name)\s*[：:]\s*"
        r"([一-龥]{2,4}|[A-Za-z][A-Za-z.'-]*(?:\s+[A-Za-z][A-Za-z.'-]*){0,2})",
        text,
        re.I,
    )
    if match:
        name = match.group(1)
    else:
        for line in lines[:10]:
            if _looks_like_name(line):
                name = line
                break
    if not name:
        for line in lines[:10]:
            if _looks_like_latin_name(line):
                name = _CONTACT_LABEL.sub("", line).strip(" ,;|")
                break
    if not name:
        return None

    if name.isascii():
        # 英文简历：取 "Example University, B.Eng. ..." 里逗号前的机构名
        for line in lines[:15]:
            cleaned = _CONTACT_LABEL.sub("", line).strip(" ：:，,;|")
            if not re.search(r"(?:University|College|Institute|Department|School)", cleaned, re.I):
                continue
            candidate = cleaned.split(",")[0].strip()
            candidate = re.split(r"\s*\d", candidate)[0].strip(" ,;|-")
            if candidate and len(candidate) <= max_affiliation:
                return f"{name}　{candidate}"
        return name

    for line in lines[:15]:
        if line == name:
            continue
        cleaned = re.sub(r"[（(][^）)]*[）)]", "", line).strip(" ：:，,")
        if re.search(r"\d", cleaned):
            continue
        if ("大学" in cleaned or "学院" in cleaned) and any(
            key in cleaned for key in ("系", "学院", "专业")
        ) and len(cleaned) <= max_affiliation:
            return f"{name}　{cleaned}"
    # 没有明确的院系行时，退一步用"XX大学/XX中心"这类单位名
    for line in lines[:15]:
        if line == name:
            continue
        cleaned = re.sub(r"[（(][^）)]*[）)]", "", line).strip(" ：:，,")
        if re.search(r"\d", cleaned):
            continue
        if ("大学" in cleaned or "学院" in cleaned) and len(cleaned) <= max_affiliation:
            return f"{name}　{cleaned}"
    return name


def extract_docx_text(data: bytes) -> tuple[str, list[str]]:
    """抽取 DOCX 正文与表格单元格。"""
    try:
        from docx import Document
    except ImportError:
        return "", ["未安装 python-docx，无法解析 DOCX"]

    try:
        document = Document(io.BytesIO(data))
    except Exception as error:  # noqa: BLE001 - 损坏文件不应让界面崩溃
        return "", [f"DOCX 打开失败：{type(error).__name__}"]

    parts = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                text = cell.text.strip()
                if text:
                    parts.append(text)
    return "\n".join(parts), []


def redact_material(text: str) -> tuple[str, list[str]]:
    """发给模型前去掉姓名行与联系方式。只处理文本，不保存原文。"""
    lines: list[str] = []
    removed = 0
    for line_no, raw in enumerate(text.splitlines()):
        line = raw.rstrip()
        if _NAME_LABEL.match(line.strip()):
            removed += 1
            continue
        if _CONTACT_LABEL.search(line):
            line = _CONTACT_LABEL.sub("", line).strip(" ：:，,;|")
            removed += 1
        if not line.strip():
            continue
        if _looks_like_name(line.strip()):
            removed += 1
            continue
        if line_no < 10 and _looks_like_latin_name(line.strip()):
            removed += 1
            continue
        if _CONTACT.search(line):
            line = _CONTACT.sub("", line)
            removed += 1
        lines.append(line)
    warnings = [f"{removed} 处姓名/联系方式已移除"] if removed else []
    return "\n".join(lines), warnings


def upload_token(name: str, size: int, raw: bytes) -> str:
    """上传文件的去重键：同名同大小但内容不同也要重新解析。"""
    return f"{name}:{size}:{hashlib.sha1(raw).hexdigest()[:12]}"


def decode_material(raw: bytes) -> str:
    """优先 UTF-8，失败再按 GB18030（兼容中文 Windows 记事本）。"""
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


__all__ = [
    "decode_material",
    "extract_docx_text",
    "extract_pdf_text",
    "extract_signature",
    "redact_material",
    "statements_from_text",
    "upload_token",
]
