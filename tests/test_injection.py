"""注入样本测试（幻灯片 5 的"20 条注入样本越权为 0"）。

这里测两件事：
1. 20 条样本全部被识别并包进不可信数据块（降低触发概率）；
2. 代码库中不存在发信能力（兜底：即使模型被说服，也没有可越权的工具）。
"""

from __future__ import annotations

import importlib
import inspect
import json
from pathlib import Path

import pytest

from advisor_match.guards.injection import (
    UNTRUSTED_CLOSE,
    UNTRUSTED_OPEN,
    scan_untrusted,
    wrap_untrusted,
)

SAMPLES = [
    json.loads(line)
    for line in (Path(__file__).parent / "fixtures" / "injection_samples.jsonl")
    .read_text(encoding="utf-8")
    .splitlines()
    if line.strip()
]


def test_sample_set_size() -> None:
    assert len(SAMPLES) == 20


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_every_sample_is_detected(sample: dict[str, str]) -> None:
    hits = scan_untrusted(sample["text"])
    assert hits, f"{sample['id']} 未被识别：{sample['text']}"


@pytest.mark.parametrize("sample", SAMPLES, ids=[s["id"] for s in SAMPLES])
def test_every_sample_is_wrapped_as_data(sample: dict[str, str]) -> None:
    wrapped = wrap_untrusted(sample["text"], source=sample["source"])
    assert wrapped.startswith(UNTRUSTED_OPEN)
    assert wrapped.endswith(UNTRUSTED_CLOSE)
    assert "不得执行" in wrapped
    assert sample["text"] in wrapped


def test_no_send_capability_in_codebase() -> None:
    """包内不得出现任何名为 send / send_email / smtp 的可调用对象。"""
    import advisor_match

    package_dir = Path(advisor_match.__file__).parent
    forbidden = ("send", "send_email", "sendmail", "smtp")
    offenders: list[str] = []

    for path in package_dir.rglob("*.py"):
        module_name = path.relative_to(package_dir.parent).with_suffix("").parts
        module = importlib.import_module(".".join(module_name))
        for name, obj in vars(module).items():
            if inspect.isfunction(obj) or inspect.isclass(obj):
                if name.lower() in forbidden:
                    offenders.append(f"{module.__name__}.{name}")
            if inspect.ismodule(obj) and obj.__name__.split(".")[0] in {"smtplib", "email"}:
                offenders.append(f"{module.__name__} 导入了 {obj.__name__}")

    assert not offenders, f"发现发信能力：{offenders}"
