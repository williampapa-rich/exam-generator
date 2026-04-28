"""
HWPX 파일 언팩 + XML 파싱

.hwpx 는 ZIP 컨테이너이며, 내부에:
  - mimetype                       (압축 안 됨)
  - META-INF/container.xml         (루트 문서 경로)
  - META-INF/manifest.xml          (파일 목록)
  - Contents/content.hpf           (패키지 메타)
  - Contents/header.xml            (스타일 정의)
  - Contents/section0.xml ~ N.xml  (실제 본문)
  - Preview/PrvText.txt            (미리보기)
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET


# ─── HWPX XML 네임스페이스 ────────────────────────────────────────────────────
NS = {
    "hh": "http://www.hancom.co.kr/hwpml/2011/head",
    "hp": "http://www.hancom.co.kr/hwpml/2011/paragraph",
    "hs": "http://www.hancom.co.kr/hwpml/2011/section",
    "hm": "http://www.hancom.co.kr/hwpml/2011/master-page",
    "ha": "http://www.hancom.co.kr/hwpml/2011/app",
    "opf": "http://www.idpf.org/2007/opf/",
    "dc":  "http://purl.org/dc/elements/1.1/",
    "manifest": "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0",
}

# ElementTree에 등록 (find/findall에서 prefix 사용 가능)
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


@dataclass
class HwpxFile:
    """언팩된 HWPX 파일의 트리 표현."""
    path: Path
    raw_bytes: dict[str, bytes] = field(default_factory=dict)  # filename → bytes
    header:    Optional[ET.Element] = None
    sections:  list[ET.Element] = field(default_factory=list)
    masterpages: list[ET.Element] = field(default_factory=list)   # masterpage0.xml ...
    content_hpf: Optional[ET.Element] = None
    manifest:    Optional[ET.Element] = None

    # ─── 편의 메서드 ──────────────────────────────────────────────────────────
    def find_in_header(self, xpath: str) -> Optional[ET.Element]:
        if self.header is None:
            return None
        return self.header.find(xpath, NS)

    def findall_in_header(self, xpath: str) -> list[ET.Element]:
        if self.header is None:
            return []
        return self.header.findall(xpath, NS)

    def all_section_paragraphs(self) -> list[ET.Element]:
        """모든 섹션의 hp:p 단락을 순서대로 반환"""
        out: list[ET.Element] = []
        for sec in self.sections:
            out.extend(sec.findall(".//hp:p", NS))
        return out


def _parse_xml_safely(data: bytes) -> Optional[ET.Element]:
    """XML 파싱 실패해도 None 반환 (부분 손상 파일 허용)"""
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        print(f"  ⚠️  XML 파싱 실패: {e}")
        return None


def unpack(hwpx_path: str | Path) -> HwpxFile:
    """
    .hwpx 파일 열기 + 모든 XML 파싱.

    Returns:
        HwpxFile 객체. 실패 시 빈 객체에 raw_bytes만 채움.
    """
    path = Path(hwpx_path)
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    f = HwpxFile(path=path)

    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            f.raw_bytes[name] = zf.read(name)

    # header.xml
    if "Contents/header.xml" in f.raw_bytes:
        f.header = _parse_xml_safely(f.raw_bytes["Contents/header.xml"])

    # section0.xml, section1.xml, ...
    section_names = sorted(
        n for n in f.raw_bytes if n.startswith("Contents/section") and n.endswith(".xml")
    )
    for sn in section_names:
        elem = _parse_xml_safely(f.raw_bytes[sn])
        if elem is not None:
            f.sections.append(elem)

    # masterpage0.xml, masterpage1.xml, ...
    mp_names = sorted(
        n for n in f.raw_bytes if "masterpage" in n.lower() and n.endswith(".xml")
    )
    for mn in mp_names:
        elem = _parse_xml_safely(f.raw_bytes[mn])
        if elem is not None:
            f.masterpages.append(elem)

    # content.hpf
    if "Contents/content.hpf" in f.raw_bytes:
        f.content_hpf = _parse_xml_safely(f.raw_bytes["Contents/content.hpf"])

    # manifest
    if "META-INF/manifest.xml" in f.raw_bytes:
        f.manifest = _parse_xml_safely(f.raw_bytes["META-INF/manifest.xml"])

    return f


def list_files(hwpx_path: str | Path) -> list[str]:
    """.hwpx 내부 파일 목록만 빠르게 조회"""
    with zipfile.ZipFile(hwpx_path, "r") as zf:
        return zf.namelist()


def get_paragraph_text(p: ET.Element) -> str:
    """hp:p 단락 → 평문 텍스트 (모든 텍스트 노드 수집).

    itertext()는 자식·손자 요소의 텍스트와 tail 텍스트까지 모두 포함하므로
    <hp:t> 내부에 컨트롤 요소가 끼어 있어도 안전하게 평문화됩니다.
    """
    return "".join(p.itertext())


def get_paragraph_props(p: ET.Element) -> dict:
    """hp:p 단락의 주요 속성 추출"""
    return {
        "id":           p.get("id"),
        "paraPrIDRef":  p.get("paraPrIDRef"),
        "styleIDRef":   p.get("styleIDRef"),
        "pageBreak":    p.get("pageBreak"),
        "columnBreak":  p.get("columnBreak"),
        "merged":       p.get("merged"),
    }


def has_table(p: ET.Element) -> bool:
    """단락이 표(hp:tbl) 컨트롤을 포함하는지 (= 지문 박스인지)"""
    return p.find(".//hp:tbl", NS) is not None


def get_table_inner_paragraphs(p: ET.Element) -> list[ET.Element]:
    """단락 내 표 안의 단락들 추출 (지문 박스 안의 본문)"""
    return p.findall(".//hp:tbl//hp:cell//hp:p", NS)
