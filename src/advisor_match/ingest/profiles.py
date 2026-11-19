"""批量抓取导师详情页，产出 ``name -> ProfileInfo``。

只抓名录里给出的系内详情页，不跟随到课题组主页（那是下一步，且有各自的
robots 与版权要求）。抓取纪律复用 ``advisors.fetch_page``：1 秒间隔、
原始 HTML 落盘、遇到验证页立即停止且不重试。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from urllib.error import HTTPError, URLError

from advisor_match.ingest.advisors import (
    BlockedError,
    ProfileInfo,
    cache_path,
    fetch_page,
    load_advisors,
    parse_profile,
)


def save_profiles(profiles: dict[str, ProfileInfo], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: asdict(profile) for name, profile in profiles.items()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def load_profiles(path: Path) -> dict[str, ProfileInfo]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: ProfileInfo(
            name=str(item.get("name") or name),
            research_areas=list(item.get("research_areas") or []),
            research_summary=item.get("research_summary"),
            homepage=item.get("homepage"),
            publications=list(item.get("publications") or []),
        )
        for name, item in payload.items()
    }


def fetch_profiles(
    advisors: list,
    *,
    cache_dir: Path | None = None,
    limit: int | None = None,
    cache_only: bool = False,
) -> tuple[dict[str, ProfileInfo], list[tuple[str, str]]]:
    """按导师列表抓详情页，返回（画像字典, [(姓名, 失败原因)]）。

    ``cache_only`` 时不发请求，只重解析本地缓存——解析器修复后不必重新抓站。
    没有 ``profile_url`` 的导师直接跳过，记入失败列表，不编造画像。
    """
    profiles: dict[str, ProfileInfo] = {}
    failures: list[tuple[str, str]] = []
    targets = [advisor for advisor in advisors if advisor.profile_url]
    if limit is not None:
        targets = targets[:limit]

    for advisor in targets:
        if cache_only:
            if cache_dir is None:
                failures.append((advisor.name, "cache_only requires cache_dir"))
                continue
            path = cache_path(cache_dir, advisor.profile_url)
            if not path.exists():
                failures.append((advisor.name, "cache missing"))
                continue
            html = path.read_text(encoding="utf-8")
        else:
            try:
                html = fetch_page(advisor.profile_url, cache_dir=cache_dir)
            except BlockedError as error:
                failures.append((advisor.name, f"blocked: {error}"))
                break  # 反爬出现时停止整批，不换 IP 重试
            except (HTTPError, URLError, TimeoutError, OSError) as error:
                failures.append((advisor.name, f"fetch error: {error}"))
                continue

        profile = parse_profile(html)
        if not profile.name:
            failures.append((advisor.name, "empty profile"))
            continue
        profiles[advisor.name] = profile

    missing = [advisor.name for advisor in advisors if not advisor.profile_url]
    failures.extend((name, "no profile_url") for name in missing)
    return profiles, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="抓取导师详情页")
    parser.add_argument("--advisors", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true", help="只重解析本地缓存，不联网")
    args = parser.parse_args(argv)

    advisors = load_advisors(args.advisors)
    profiles, failures = fetch_profiles(
        advisors,
        cache_dir=args.cache_dir or args.out.parent / "raw_profiles",
        limit=args.limit,
        cache_only=args.cache_only,
    )
    save_profiles(profiles, args.out)

    print(f"profiles: {len(profiles)}")
    print(f"failures: {len(failures)}")
    for name, reason in failures[:20]:
        print(f"  {name}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
