"""界面冒烟测试：真实数据 → 匹配 → 草稿 → 留痕，且没有发送入口。"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from advisor_match.schemas import AdmissionEntry, Advisor, SourceRef

APP = Path(__file__).parents[1] / "app" / "streamlit_app.py"


def admission(direction: str) -> AdmissionEntry:
    return AdmissionEntry(
        year=2027,
        program_code="081200",
        program_name="计算机科学与技术",
        program_type="学术学位",
        direction_code="01",
        direction=direction,
        source="https://yzbm.example.edu.cn/catalog",
    )


@pytest.fixture()
def advisors_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    advisors = [
        Advisor(
            name="李老师",
            email="li@example.edu.cn",
            research_tags=["信息检索", "多模态学习"],
            research_summary="研究信息检索与多模态学习。",
            fetched_at=date(2026, 11, 21),
            admissions=[admission("信息检索")],
            profile_url="https://www.cs.example.edu.cn/info/li.htm",
            sources=[SourceRef(doc_id="p-li", locator="研究领域", quote="信息检索，多模态学习")],
        ),
        Advisor(
            name="王老师",
            email="wang@example.edu.cn",
            research_tags=["数据库", "数据管理"],
            research_summary="研究数据库系统。",
            fetched_at=date(2026, 11, 21),
            admissions=[admission("数据工程与知识工程")],
            sources=[SourceRef(doc_id="p-wang", locator="研究领域", quote="数据库，数据管理")],
        ),
        Advisor(
            name="赵老师",
            email=None,
            research_tags=["信息检索"],
            sources=[SourceRef(doc_id="p-zhao", locator="研究领域", quote="信息检索")],
        ),
    ]
    path = tmp_path / "advisors.jsonl"
    path.write_text(
        "\n".join(advisor.model_dump_json() for advisor in advisors) + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("AM_ADVISORS_PATH", str(path))
    monkeypatch.setenv("AM_CONFIRMATIONS_PATH", str(tmp_path / "confirmations.jsonl"))
    monkeypatch.setenv("AM_DISABLE_VECTOR", "1")
    monkeypatch.setenv("AM_DISABLE_LLM", "1")
    return path


def goto(app: AppTest, label: str) -> AppTest:
    app.radio(key="section").set_value(label).run()
    return app


def run_matching_flow(
    app: AppTest, target: str = "信息检索", statement: str = "完成信息检索方向的课程项目"
) -> AppTest:
    # 测试里关掉向量模型，避免加载 95MB 模型
    app.run()
    app.text_input(key="targets").set_value(target)
    app.text_area(key="statements").set_value(statement)
    app.button(key="run_matching").click().run()
    return goto(app, "② 导师匹配")


def test_app_renders_without_exception(advisors_file: Path) -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    assert app.radio(key="section").options == ["① 材料与台账", "② 导师匹配", "③ 邮件草稿"]
    # 首屏不加载导师索引，提示与页面同时出现
    assert any("开始匹配" in info.value for info in app.info)


def test_matching_shows_tiers_and_provenance(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("核心契合" in item.value for item in app.markdown)
    # 召回/重排的内部计数不再展示给学生
    assert not any(caption.startswith(("[recall]", "[rerank]", "[tiers]")) for caption in captions)
    assert not any("索引导师" in caption for caption in captions)
    # 招生状态只写资格，不写名额
    assert any("名额未公布" in caption for caption in captions)
    assert any("信息截止" in caption for caption in captions)


def test_generating_draft_shows_evidence_and_confirmation(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("李老师").run()
    app.text_input(key="signature").set_value("张三　计算机系 2023 级")
    app.button(key="generate_draft").click().run()

    assert not app.exception
    captions = [caption.value for caption in app.caption]
    assert any("校验：保留" in caption for caption in captions)
    assert any(caption.startswith("依据：") for caption in captions)

    app.button(key="confirm_draft").click().run()

    assert not app.exception
    assert any("已留痕" in message.value for message in app.toast)
    log = Path(os.environ["AM_CONFIRMATIONS_PATH"])
    assert log.exists() and log.read_text(encoding="utf-8").strip()


def test_sentence_editing_and_deletion(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("李老师").run()
    app.button(key="generate_draft").click().run()

    editors = [item for item in app.text_area if item.key and item.key.startswith("edit-")]
    assert len(editors) >= 4
    editors[0].set_value("我改写后的第一句。")
    deleters = [item for item in app.checkbox if item.key and item.key.startswith("del-")]
    deleters[1].check()
    app.run()
    app.button(key="confirm_draft").click().run()

    assert not app.exception
    values = [item.value for item in app.text_area if item.key and item.key.startswith("edit-")]
    assert "我改写后的第一句。" in values
    assert len(values) == len(editors) - 1

    assert any("已留痕" in message.value for message in app.toast)
    log = Path(os.environ["AM_CONFIRMATIONS_PATH"])
    record = json.loads(log.read_text(encoding="utf-8").splitlines()[-1])
    assert record["student_edit_ratio"] > 0


def test_switching_advisors_keeps_each_draft(advisors_file: Path) -> None:
    app = run_matching_flow(
        AppTest.from_file(str(APP), default_timeout=30).run(),
        target="信息检索,数据库",
        statement="完成信息检索与数据库方向的课程项目",
    )
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("李老师").run()
    app.button(key="generate_draft").click().run()
    assert any("推免自荐｜李老师" in item.value for item in app.markdown)

    # 切到王老师：还没有草稿，不应显示李老师的草稿
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("王老师").run()
    editors = [item for item in app.text_area if item.key and item.key.startswith("edit-")]
    assert editors == []
    app.button(key="generate_draft").click().run()
    assert any("推免自荐｜王老师" in item.value for item in app.markdown)

    # 切回李老师：草稿还在
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("李老师").run()
    assert any("推免自荐｜李老师" in item.value for item in app.markdown)
    assert [item for item in app.text_area if item.key and item.key.startswith("edit-")]


def test_regenerating_draft_resets_widget_state(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    app.button(key="generate_draft").click().run()

    editors = [item for item in app.text_area if item.key and item.key.startswith("edit-")]
    editors[0].set_value("我改过的句子。")
    app.run()

    # 重新生成同一导师的草稿：旧的编辑不能带过来
    app.button(key="generate_draft").click().run()
    values = [item.value for item in app.text_area if item.key and item.key.startswith("edit-")]
    assert "我改过的句子。" not in values


def test_clear_draft_removes_only_that_advisor(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("李老师").run()
    app.button(key="generate_draft").click().run()
    assert [item for item in app.text_area if item.key and item.key.startswith("edit-")]

    app.button(key="clear_draft").click().run()
    assert not [item for item in app.text_area if item.key and item.key.startswith("edit-")]
    assert not any("已保存草稿" in caption.value for caption in app.caption)


def test_missing_email_blocks_draft(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    app.selectbox(key="advisor_select").set_value("赵老师").run()
    app.button(key="generate_draft").click().run()
    assert any("邮箱未收录" in error.value for error in app.error)


def test_missing_resume_and_statements_shows_error(advisors_file: Path) -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.run()
    app.button(key="run_matching").click().run()
    assert any("请先上传简历" in error.value for error in app.error)


def test_target_direction_is_optional(advisors_file: Path) -> None:
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.run()
    app.text_area(key="statements").set_value("完成信息检索方向的课程项目")
    app.button(key="run_matching").click().run()
    assert not app.exception
    assert not any("目标方向" in error.value for error in app.error)


def test_no_match_shows_warning(advisors_file: Path) -> None:
    app = run_matching_flow(
        AppTest.from_file(str(APP), default_timeout=30).run(),
        target="量子计算",
        statement="完成量子计算方向的课程项目",
    )
    assert not app.exception
    assert any("召回没有候选" in info.value for info in app.info)
    goto(app, "③ 邮件草稿")
    assert any("当前没有可写的导师" in warning.value for warning in app.warning)


def test_ui_has_no_send_button(advisors_file: Path) -> None:
    app = run_matching_flow(AppTest.from_file(str(APP), default_timeout=30).run())
    goto(app, "③ 邮件草稿")
    labels = [button.label for button in app.button]
    assert "开始匹配" in labels and "生成草稿" in labels
    assert all("发送" not in label for label in labels)


def make_pdf(path: Path, lines: list[str] | None = None) -> bytes:
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for index, line in enumerate(lines or []):
        page.insert_text((50, 60 + index * 18), line, fontsize=10)
    doc.save(path)
    doc.close()
    return path.read_bytes()


def upload(app: AppTest, name: str, content: bytes) -> AppTest:
    app.file_uploader(key="materials_upload").set_value(
        (name, content, "application/pdf")
    )
    app.run()
    return app


def test_uploaded_pdf_fills_statements(advisors_file: Path, tmp_path: Path) -> None:
    content = make_pdf(tmp_path / "resume.pdf", ["Information retrieval course project"])
    app = upload(AppTest.from_file(str(APP), default_timeout=30).run(), "resume.pdf", content)

    assert not app.exception
    assert "Information retrieval course project" in app.text_area(key="statements").value
    assert any("按行解析" in caption.value for caption in app.caption)


def test_new_upload_replaces_ledger_and_invalidates_match(
    advisors_file: Path, tmp_path: Path
) -> None:
    app = run_matching_flow(
        AppTest.from_file(str(APP), default_timeout=30).run(),
        target="信息检索",
        statement="完成信息检索方向的课程项目",
    )
    assert app.session_state["result"] is not None

    content = make_pdf(tmp_path / "new.pdf", ["Database course project"])
    upload(app, "new.pdf", content)

    # 新材料立刻替换台账，旧的推荐与草稿作废，等学生重新匹配
    assert "Database course project" in app.text_area(key="statements").value
    goto(app, "① 材料与台账")
    assert any("Database course project" in item.value for item in app.markdown)
    goto(app, "② 导师匹配")
    assert any("材料已更新" in info.value for info in app.info)


def test_pdf_without_text_names_the_file(advisors_file: Path, tmp_path: Path) -> None:
    content = make_pdf(tmp_path / "scan.pdf")
    app = upload(AppTest.from_file(str(APP), default_timeout=30).run(), "scan.pdf", content)
    app.button(key="run_matching").click().run()

    assert not app.exception
    assert any(
        "已收到《scan.pdf》" in error.value and "扫描版" in error.value
        for error in app.error
    )


def test_parse_failure_is_reported(
    advisors_file: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import advisor_match.product.materials as materials

    def boom(data: bytes) -> tuple[str, list[str]]:
        raise ValueError("boom")

    monkeypatch.setattr(materials, "extract_pdf_text", boom)
    content = make_pdf(tmp_path / "resume.pdf", ["Information retrieval course project"])
    app = upload(AppTest.from_file(str(APP), default_timeout=30).run(), "resume.pdf", content)

    assert not app.exception
    assert any("解析失败" in caption.value for caption in app.caption)
    app.button(key="run_matching").click().run()
    assert any("解析失败" in error.value for error in app.error)


def test_missing_data_falls_back_to_demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AM_ADVISORS_PATH", str(tmp_path / "missing.jsonl"))
    monkeypatch.setenv("AM_DISABLE_VECTOR", "1")
    monkeypatch.setenv("AM_DISABLE_LLM", "1")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    app.text_input(key="targets").set_value("信息检索")
    app.text_area(key="statements").set_value("完成信息检索方向的课程项目")
    app.button(key="run_matching").click().run()
    assert not app.exception
    assert any("虚构导师演示" in warning.value for warning in app.warning)
