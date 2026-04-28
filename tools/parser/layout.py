"""
section0.xml 의 hp:secPr 에서 페이지 레이아웃 추출.

추출 대상:
  - 페이지 크기 + 여백
  - 단 구조 (단 수, 간격)
  - 헤더/푸터 영역 길이 + 안 내용
  - masterPage (헤더/푸터의 실제 텍스트/필드)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import xml.etree.ElementTree as ET

from tools.parser.unpack import HwpxFile, NS, get_paragraph_text


@dataclass
class PageSpec:
    width:        int = 59528    # A4 기본
    height:       int = 84188
    margin_left:  int = 8504
    margin_right: int = 8504
    margin_top:   int = 5669
    margin_bot:   int = 4252
    header_len:   int = 4252
    footer_len:   int = 2835
    gutter_len:   int = 0


@dataclass
class ColumnSpec:
    count:    int = 1
    same_size: bool = True
    gap:      int = 0
    type:     str = "NEWSPAPER"
    layout:   str = "LEFT"


@dataclass
class HeaderFooterContent:
    """헤더 또는 푸터 안의 실제 텍스트 + 필드"""
    type:      str = "HEADER"            # HEADER / FOOTER
    text:      str = ""                  # 평문 합본
    has_pagenum: bool = False            # 페이지 번호 필드 포함 여부
    raw_xml:    str = ""                 # 원본 XML 문자열


@dataclass
class LayoutSpec:
    page:    PageSpec = field(default_factory=PageSpec)
    columns: ColumnSpec = field(default_factory=ColumnSpec)
    header:  Optional[HeaderFooterContent] = None
    footer:  Optional[HeaderFooterContent] = None
    raw_secpr: str = ""

    def summary(self) -> str:
        return (
            f"A4{'2단' if self.columns.count == 2 else f'{self.columns.count}단'} · "
            f"여백 L{self.page.margin_left} R{self.page.margin_right} "
            f"T{self.page.margin_top} B{self.page.margin_bot} · "
            f"H{'있음' if self.header else '없음'} F{'있음' if self.footer else '없음'}"
        )


# ─── 추출 함수 ─────────────────────────────────────────────────────────────────

def _extract_page(secpr) -> PageSpec:
    pp = secpr.find("hp:pagePr", NS) if secpr is not None else None
    if pp is None:
        return PageSpec()
    return PageSpec(
        width=int(pp.get("paperWidth", "59528")),
        height=int(pp.get("paperHeight", "84188")),
        margin_left=int(pp.get("leftMargin", "8504")),
        margin_right=int(pp.get("rightMargin", "8504")),
        margin_top=int(pp.get("topMargin", "5669")),
        margin_bot=int(pp.get("bottomMargin", "4252")),
        header_len=int(pp.get("headerLen", "4252")),
        footer_len=int(pp.get("footerLen", "2835")),
        gutter_len=int(pp.get("gutterLen", "0")),
    )


def _extract_columns(secpr) -> ColumnSpec:
    cp = secpr.find("hp:colPr", NS) if secpr is not None else None
    if cp is None:
        return ColumnSpec()
    return ColumnSpec(
        count=int(cp.get("colCount", "1")),
        same_size=cp.get("sameSz", "1") == "1",
        gap=int(cp.get("sameGap", "0")),
        type=cp.get("type", "NEWSPAPER"),
        layout=cp.get("layout", "LEFT"),
    )


def _extract_header_footer(secpr, kind: str) -> Optional[HeaderFooterContent]:
    """kind = 'HEADER' or 'FOOTER'"""
    if secpr is None:
        return None
    master = secpr.find("hp:masterPage", NS)
    if master is None:
        return None
    for hf in master.findall("hp:headerFooter", NS):
        if hf.get("type") == kind:
            paragraphs = hf.findall(".//hp:p", NS)
            text = " ".join(get_paragraph_text(p).strip() for p in paragraphs if get_paragraph_text(p).strip())
            has_pagenum = hf.find(".//hp:pageNum", NS) is not None
            raw = ET.tostring(hf, encoding="unicode")
            return HeaderFooterContent(
                type=kind, text=text, has_pagenum=has_pagenum, raw_xml=raw,
            )
    return None


def extract_layout(hwpx: HwpxFile) -> LayoutSpec:
    """첫 번째 섹션의 hp:secPr 에서 레이아웃 정보 추출"""
    if not hwpx.sections:
        return LayoutSpec()

    sec = hwpx.sections[0]
    secpr = sec.find(".//hp:secPr", NS)
    raw = ET.tostring(secpr, encoding="unicode") if secpr is not None else ""

    layout = LayoutSpec(
        page=_extract_page(secpr),
        columns=_extract_columns(secpr),
        header=_extract_header_footer(secpr, "HEADER"),
        footer=_extract_header_footer(secpr, "FOOTER"),
        raw_secpr=raw,
    )

    # 평가원 양식: 마스터페이지가 별도 파일(masterpage0.xml 등)에 있을 수 있음
    # secPr에서 못 찾았으면 별도 파일에서 검색
    if layout.header is None or layout.footer is None:
        for mp in hwpx.masterpages:
            for hf in mp.findall(".//hp:headerFooter", NS):
                kind = hf.get("type")
                if kind == "HEADER" and layout.header is None:
                    paragraphs = hf.findall(".//hp:p", NS)
                    text = " ".join(get_paragraph_text(p).strip() for p in paragraphs if get_paragraph_text(p).strip())
                    has_pn = hf.find(".//hp:pageNum", NS) is not None
                    layout.header = HeaderFooterContent(
                        type="HEADER", text=text[:200], has_pagenum=has_pn,
                        raw_xml=ET.tostring(hf, encoding="unicode"),
                    )
                elif kind == "FOOTER" and layout.footer is None:
                    paragraphs = hf.findall(".//hp:p", NS)
                    text = " ".join(get_paragraph_text(p).strip() for p in paragraphs if get_paragraph_text(p).strip())
                    has_pn = hf.find(".//hp:pageNum", NS) is not None
                    layout.footer = HeaderFooterContent(
                        type="FOOTER", text=text[:200], has_pagenum=has_pn,
                        raw_xml=ET.tostring(hf, encoding="unicode"),
                    )

    # 단(column) 정보도 마스터페이지에 있을 수 있음
    if layout.columns.count == 1:
        for mp in hwpx.masterpages:
            cp = mp.find(".//hp:colPr", NS)
            if cp is not None:
                layout.columns = ColumnSpec(
                    count=int(cp.get("colCount", "1")),
                    same_size=cp.get("sameSz", "1") == "1",
                    gap=int(cp.get("sameGap", "0")),
                    type=cp.get("type", "NEWSPAPER"),
                    layout=cp.get("layout", "LEFT"),
                )
                if layout.columns.count > 1:
                    break

    return layout
