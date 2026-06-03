"""한글 HWPX(.hwpx) 파서

표준 라이브러리(zipfile + xml.etree)만 사용하여 OWPML(zip+XML) 구조의
.hwpx 파일에서 텍스트를 추출합니다. 외부 의존성이 없습니다.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


def _localname(tag: str) -> str:
    """``{ns}local`` 형태의 태그에서 네임스페이스를 제거한 로컬명을 반환한다.

    Args:
        tag: ElementTree 가 돌려주는 정규화된 태그명.

    Returns:
        네임스페이스를 제거한 로컬 태그명.
    """
    return tag.rsplit("}", 1)[-1]


class HwpxParser(DocumentParser):
    """한글 HWPX(.hwpx) 파일 파서"""

    extensions = [".hwpx"]

    def parse(self, path: Path) -> str:
        """.hwpx 파일에서 텍스트를 추출한다.

        OWPML 패키지(zip) 내부의 ``Contents/section*.xml`` 을 순서대로 순회하며
        문단(로컬명 ``p``)별 텍스트 런(로컬명 ``t``)을 모은다. 네임스페이스
        접두어가 버전마다 달라질 수 있어 로컬 태그명으로 매칭한다.

        Args:
            path: 파싱할 .hwpx 파일 경로.

        Returns:
            추출된 텍스트. 추출 실패 시 빈 문자열.
        """
        try:
            with zipfile.ZipFile(path) as zf:
                section_names = sorted(
                    name
                    for name in zf.namelist()
                    if name.startswith("Contents/section") and name.endswith(".xml")
                )
                parts = [self._extract_section_text(zf.read(name)) for name in section_names]
            return "\n\n".join(p for p in parts if p)
        except Exception as e:
            logger.warning("hwpx_parse_failed", path=str(path), error=str(e))
            return ""

    @staticmethod
    def _extract_section_text(xml_bytes: bytes) -> str:
        """section XML 바이트에서 문단 텍스트를 추출한다.

        문단(로컬명 ``p``) 단위로 그 안의 텍스트 런(로컬명 ``t``)을 이어붙여
        한 줄로 만들고, 문단마다 줄바꿈으로 구분한다.

        Args:
            xml_bytes: ``section*.xml`` 의 원본 바이트.

        Returns:
            추출된 문단 텍스트(줄바꿈 구분). 파싱 실패 시 빈 문자열.
        """
        try:
            root = ET.fromstring(xml_bytes)
        except ET.ParseError:
            return ""

        lines: list[str] = []
        for para in root.iter():
            if _localname(para.tag) != "p":
                continue
            runs = [
                elem.text
                for elem in para.iter()
                if _localname(elem.tag) == "t" and elem.text
            ]
            line = "".join(runs).strip()
            if line:
                lines.append(line)
        return "\n".join(lines)
