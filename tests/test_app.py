"""界面冒烟测试：真实数据 → 匹配 → 草稿，且没有发送入口。"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from advisor_match.schemas import Advisor, SourceRef

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


@pytest.fixture()
def advisors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    advisors = [
        Advisor(
            name="李老师",
            email="li@example.edu.cn",
            research_tags=["信息检索", "多模态学习"],
            research_summary="研究信息检索与多模态学习。",
            sources=[SourceRef(doc_id="p-li", locator="研究领域", quote="信息检索，多模态学习")],
        ),
        Advisor(
            name="王老师",
            email="wang@example.edu.cn",
            research_tags=["数据库", "数据管理"],
            research_summary="研究数据库系统。",
            sources=[SourceRef(doc_id="p-wang", locator="研究领域", quote="数据库，数据管理")],
        ),
    ]
    path = tmp_path / "advisors.jsonl"
    path.write_text(
        "\n".join(advisor.model_dump_json() for advisor in advisors) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("AM_ADVISORS_PATH", str(path))
    return path


def run_matching_flow(app: AppTest) -> AppTest:
    # 测试里关掉向量模型，避免加载 95MB 模型
    app.checkbox(key="use_vector").uncheck().run()
    app.text_input(key="targets").set_value("信息检索")
    app.text_area(key="statements").set_value("完成信息检索方向的课程项目")
    app.button(key="run_matching").click().run()
    return app


def test_app_renders_without_exception(advisors_file: Path) -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["① 材料与台账", "② 导师匹配", "③ 邮件草稿"]


def test_matching_shows_tiers(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("索引导师 2 位" in caption for caption in captions)
    assert any("核心契合" in item.value for item in app.markdown)


def test_generating_draft_shows_evidence_and_confirmation(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    app.selectbox(key="advisor_select").set_value("李老师").run()
    app.text_input(key="signature").set_value("张三　计算机系 2023 级")
    app.button(key="generate_draft").click().run()

    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("校验：保留" in caption for caption in captions)
    assert any(caption.startswith("依据：") for caption in captions)
    assert sum(1 for box in app.checkbox if box.label == "已核对") >= 4


def test_ui_has_no_send_button(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    labels = [button.label for button in app.button]
    assert "开始匹配" in labels and "生成草稿" in labels
    assert all("发送" not in label for label in labels)


def test_missing_data_falls_back_to_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AM_ADVISORS_PATH", str(tmp_path / "missing.jsonl"))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.checkbox(key="use_vector").uncheck().run()
    app.text_input(key="targets").set_value("信息检索")
    app.text_area(key="statements").set_value("完成信息检索方向的课程项目")
    app.button(key="run_matching").click().run()
    assert not app.exception
    assert any("虚构导师演示" in warning.value for warning in app.warning)
