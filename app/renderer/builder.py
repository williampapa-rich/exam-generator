"""
HWPX 빌더 — templates/template.hwpx 를 base로 placeholder 치환 후 ZIP 재패키징.

base 템플릿 = parser SSOT (template_fixed.hwpx).
section0.xml + masterpage*.xml + Preview/PrvText.txt 만 동적 인젝션.
나머지 파일은 그대로 캐시 → 매 요청마다 동일.
"""
from __future__ import annotations

import io
import zipfile

from app.renderer.template_injector import (
    get_template_files, render_masterpage, render_section0,
)
from shared.schemas.question import ExamMeta, Question

_MIMETYPE = b"application/hwp+zip"


def _preview_text(questions: list[Question]) -> str:
    return "\n".join(
        f"{i+1}. {q.question_text}" for i, q in enumerate(questions)
    )


def build_hwpx(meta: ExamMeta, questions: list[Question]) -> io.BytesIO:
    """완성된 .hwpx 파일을 BytesIO로 반환."""
    section0_xml = render_section0(meta, questions).encode("utf-8")
    preview_txt  = _preview_text(questions).encode("utf-8")

    overrides: dict[str, bytes] = {
        "Contents/section0.xml": section0_xml,
        "Preview/PrvText.txt":   preview_txt,
    }
    # masterpage 들도 머릿글 placeholder를 위해 인젝션
    for name in get_template_files():
        if "masterpage" in name and name.endswith(".xml"):
            overrides[name] = render_masterpage(name, meta)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zout:
        # mimetype 은 STORED + 첫 번째 (OCF 규약)
        info = zipfile.ZipInfo("mimetype")
        info.compress_type = zipfile.ZIP_STORED
        zout.writestr(info, _MIMETYPE)

        for name, data in get_template_files().items():
            zout.writestr(name, overrides.get(name, data),
                          compress_type=zipfile.ZIP_DEFLATED)

    buf.seek(0)
    return buf
