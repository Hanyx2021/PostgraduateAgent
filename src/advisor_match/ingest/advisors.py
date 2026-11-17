"""导师库构建：系主页教师名录 + 个人详情页。

抓取纪律（写进代码，不只是约定）：
- 只访问公开页面；请求间隔 ≥ 1s；UA 标明用途；
- 遇到反爬验证页立即停止并报错，不换 IP、不伪装 UA；
- **不保存电话号码**：名录里虽有，但发信只需要工作邮箱，收集电话超出必要范围；
- 原始 HTML 落盘到 ``data/raw/``，保证结果可复现、可回溯。

页面结构（2020 版名录）::

    <dl><dt><h4>研究所</h4></dt>
        <dd><h3>正高</h3><ul><li>
            <h2><a href="../info/1111/3490.htm">姓名</a></h2>
            <p>职称</p><p>电话</p><p>邮箱</p>
        </li></ul></dd>
    </dl>
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from advisor_match.schemas import Advisor, SourceRef

DEPARTMENT_NAME = "清华大学计算机科学与技术系"
USER_AGENT = "AdvisorMatch/0.1 (course project; contact: see README) robots-respecting"
MIN_INTERVAL_SECONDS = 1.0
BLOCK_MARKERS = ("not a bot", "please verify you are human", "访问过于频繁", "安全验证")

_TAG_SPLIT = re.compile(r"[，,、；;/]|以及")
_last_request_at = 0.0


class BlockedError(RuntimeError):
    """目标站点返回了反爬验证页。此时应当停止，而不是绕过。"""


@dataclass(frozen=True)
class DirectoryEntry:
    """名录页的一条记录。刻意不含电话。"""

    name: str
    title: str
    email: str
    unit: str
    rank_group: str
    profile_url: str


@dataclass(frozen=True)
class ProfileInfo:
    """详情页抽取结果。"""

    name: str
    research_areas: list[str] = field(default_factory=list)
    research_summary: str | None = None
    homepage: str | None = None
    publications: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# 解析
# --------------------------------------------------------------------------
def parse_directory(html: str, *, base_url: str) -> list[DirectoryEntry]:
    """解析教职工名录页，按页面顺序返回条目。

    电话在解析阶段就被丢弃——不进入任何中间结构，避免"以后可能会用"的诱惑。
    """
    soup = BeautifulSoup(html, "html.parser")
    entries: list[DirectoryEntry] = []

    for dl in soup.select("div.people01-nr dl"):
        unit_node = dl.find("dt")
        unit = unit_node.get_text(strip=True) if unit_node else ""
        for dd in dl.find_all("dd"):
            group_node = dd.find("h3")
            rank_group = group_node.get_text(strip=True) if group_node else ""
            for li in dd.select("li"):
                link = li.select_one("h2 a")
                if link is None:
                    continue
                name = (link.get("title") or link.get_text(strip=True)).strip()
                href = link.get("href") or ""
                paragraphs = [p.get_text(strip=True) for p in li.select("div.text p")]
                title = paragraphs[0] if paragraphs else ""
                email = next((p for p in paragraphs if "@" in p), "")
                if not name:
                    continue
                entries.append(
                    DirectoryEntry(
                        name=name,
                        title=title,
                        email=email,
                        unit=unit,
                        rank_group=rank_group,
                        profile_url=urljoin(base_url, href),
                    )
                )
    return entries


def _sections(content) -> dict[str, list[str]]:
    """把正文按 <h4> 标题切成若干段。"""
    sections: dict[str, list[str]] = {}
    current = ""
    for node in content.find_all(["h4", "p"]):
        if node.name == "h4":
            current = node.get_text(strip=True)
            sections.setdefault(current, [])
            continue
        # <h4><p>标题</p></h4> 里的 p 不是正文，跳过。
        if node.find_parent("h4") is not None:
            continue
        text = node.get_text(" ", strip=True)
        # 有些页面用 <p><strong>标题</strong></p> 代替 h4。
        # 以冒号结尾的加粗段落也是标题（如"承担及完成的重要科研项目:"）。
        strong = node.find("strong")
        if strong is not None and text == strong.get_text(" ", strip=True):
            if len(text) <= 20 or text.endswith((":", "：")):
                current = text
                sections.setdefault(current, [])
                continue
        sections.setdefault(current, []).append(text)
    return sections


def parse_profile(html: str) -> ProfileInfo:
    """解析个人详情页：研究领域、主页、学术成果。"""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#vsb_content_2 .v_news_content") or soup.select_one(".v_news_content")
    if content is None:
        return ProfileInfo(name="")

    name_node = soup.select_one(".show01 h5")
    name = name_node.get_text(strip=True) if name_node else ""
    sections = _sections(content)

    areas: list[str] = []
    for line in sections.get("研究领域", []):
        for tag in _TAG_SPLIT.split(line):
            tag = tag.strip()
            # 研究方向是一个短语；长句说明标题漏识别，宁可丢弃也不当标签。
            if tag and len(tag) <= 40 and not tag.endswith((":", "：")):
                areas.append(tag)
    areas = areas[:15]

    homepage = None
    for paragraph in content.find_all("p"):
        text = paragraph.get_text(" ", strip=True)
        if "主页" not in text:
            continue
        anchor = paragraph.find("a", href=True)
        if anchor is not None:
            homepage = anchor["href"]
        else:
            match = re.search(r"https?://\S+", text)
            homepage = match.group(0) if match else None
        break

    summary_lines = [line for line in sections.get("研究概况", []) if line]
    return ProfileInfo(
        name=name,
        research_areas=areas,
        research_summary="\n".join(summary_lines) if summary_lines else None,
        homepage=homepage,
        publications=sections.get("学术成果", []),
    )


# --------------------------------------------------------------------------
# 抓取
# --------------------------------------------------------------------------
def _looks_blocked(html: str) -> bool:
    head = html[:4000].lower()
    return any(marker.lower() in head for marker in BLOCK_MARKERS)


def _throttle() -> None:
    global _last_request_at
    wait = MIN_INTERVAL_SECONDS - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def cache_path(cache_dir: Path, url: str) -> Path:
    """原始页面的本地缓存路径。同一 URL 永远落到同一文件，便于离线重解析。"""
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return cache_dir / f"{digest}.html"


def fetch_page(url: str, *, cache_dir: Path | None = None) -> str:
    """抓取一个页面，返回 HTML。命中反爬页时抛 ``BlockedError``。"""
    _throttle()
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - 仅抓取公开的系主页
        raw = response.read()
    html = raw.decode("utf-8", errors="replace")
    if _looks_blocked(html):
        raise BlockedError(f"{url} 返回了验证页，停止抓取")

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path(cache_dir, url).write_text(html, encoding="utf-8")
        with (cache_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"url": url, "fetched_at": date.today().isoformat()}, ensure_ascii=False) + "\n")
    return html


# --------------------------------------------------------------------------
# 组装
# --------------------------------------------------------------------------
def to_advisor(entry: DirectoryEntry, profile: ProfileInfo, *, fetched_at: date) -> Advisor:
    """把名录条目与详情页合成 ``Advisor``。

    研究方向的证据指向详情页的「研究领域」原文；没有抓到详情页时不编造标签。
    招生记录由 ``catalog.apply_catalog`` 另行写入。
    """
    sources: list[SourceRef] = []
    if profile.research_areas:
        sources.append(
            SourceRef(
                doc_id=entry.profile_url,
                locator="研究领域",
                quote="，".join(profile.research_areas),
            )
        )
    return Advisor(
        name=entry.name,
        affiliation=DEPARTMENT_NAME,
        unit=entry.unit or None,
        title=entry.title or None,
        email=entry.email or None,
        profile_url=entry.profile_url,
        homepage=profile.homepage,
        research_tags=profile.research_areas,
        research_summary=profile.research_summary,
        recruiting=None,
        fetched_at=fetched_at,
        sources=sources,
    )


def load_advisors(path: Path) -> list[Advisor]:
    advisors: list[Advisor] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                advisors.append(Advisor.model_validate_json(line))
    return advisors


def save_advisors(advisors: list[Advisor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for advisor in advisors:
            handle.write(advisor.model_dump_json() + "\n")


__all__ = [
    "BlockedError",
    "DirectoryEntry",
    "ProfileInfo",
    "fetch_page",
    "load_advisors",
    "parse_directory",
    "parse_profile",
    "save_advisors",
    "to_advisor",
]
