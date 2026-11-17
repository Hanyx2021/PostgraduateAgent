"""敏感属性守卫。

原则：不是"生成时不看敏感属性"，而是"数据结构里根本没有这些字段"。
前者依赖提醒，后者依赖权限与类型系统（幻灯片 5 的"关键限制由程序执行"）。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

#: 禁止出现在任何进入模型的画像里的字段名（小写、下划线）。
#: 注意这里列的是"属性名"，与具体取值无关。
SENSITIVE_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "name",
        "full_name",
        "gender",
        "sex",
        "birth",
        "birthday",
        "age",
        "hometown",
        "native_place",
        "province",
        "school",
        "university",
        "undergrad_school",
        "college",
        "major",
        "ethnicity",
        "nationality",
        "politics",
        "id_number",
        "phone",
        "wechat",
        "address",
    }
)

#: 取值层面的敏感线索：自由文本里出现的院校名、籍贯等应由脱敏器处理，
#: 而不是靠字段检查。这里是给 ``privacy/redact.py`` 的提示词。
SENSITIVE_TEXT_HINTS: tuple[str, ...] = (
    "中学",
    "高中",
    "籍贯",
    "生源地",
)


def assert_no_sensitive_fields(model: BaseModel | Any, *, path: str = "") -> None:
    """递归检查模型（及嵌套模型/列表）中是否存在敏感字段。

    发现即抛 ``ValueError``——在流水线入口调用它，比在提示词里写"请不要使用"更可靠。
    注意 ``StudentProfile`` 允许 ``constraints`` 这类自由文本，但自由文本进入模型前
    必须经过 ``privacy/redact.py`` 脱敏。
    """
    if isinstance(model, BaseModel):
        for field_name in type(model).model_fields:
            dotted = f"{path}.{field_name}" if path else field_name
            if field_name.lower() in SENSITIVE_FIELD_NAMES:
                raise ValueError(f"画像中出现敏感字段：{dotted}")
            assert_no_sensitive_fields(getattr(model, field_name), path=dotted)
    elif isinstance(model, (list, tuple)):
        for index, item in enumerate(model):
            assert_no_sensitive_fields(item, path=f"{path}[{index}]")
