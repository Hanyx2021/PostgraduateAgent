"""界面冒烟测试：页面能跑、能生成草稿、界面上没有发送按钮。"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def test_app_renders_without_exception() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["① 材料与台账", "② 导师匹配", "③ 邮件草稿"]


def test_generating_draft_shows_validation_and_sentences() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.text_input[0].set_value("张三　计算机系 2023 级")
    app.text_area[0].set_value("我做过多模态检索增强生成的课程项目")
    app.button[0].click().run()

    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("校验：保留" in caption for caption in captions)
    assert any("依据：L-0002" in caption for caption in captions)
    assert len(app.checkbox) >= 4


def test_ui_has_no_send_button() -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    labels = [button.label for button in app.button]
    assert labels == ["生成草稿"]
    assert all("发送" not in label for label in labels)
