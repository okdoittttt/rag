"""문서 파서 테스트

신규 파일 형식(docx/xlsx/pptx/csv/hwpx) 파서와 파서 레지스트리(단일 소스),
로더 위임을 검증한다. 서드파티 라이브러리가 필요한 파서는 동일 라이브러리로
샘플 파일을 생성한 뒤 round-trip 으로 확인하며, 미설치 시 ``importorskip`` 으로
건너뛴다.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from rag.ingestion.loader import DocumentLoader, load_file
from rag.ingestion.parsers import (
    CsvParser,
    HwpxParser,
    get_parser,
    get_supported_extensions,
)


class TestParserRegistry:
    """파서 레지스트리(단일 소스) 검증"""

    def test_supported_extensions_include_all_formats(self) -> None:
        """레지스트리가 신규 포맷을 모두 노출한다."""
        exts = get_supported_extensions()
        for ext in [".txt", ".md", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".hwpx"]:
            assert ext in exts

    def test_supported_extensions_have_no_duplicates(self) -> None:
        """확장자 목록에 중복이 없다."""
        exts = get_supported_extensions()
        assert len(exts) == len(set(exts))

    def test_get_parser_dispatches_by_extension(self) -> None:
        """확장자에 맞는 파서가 선택된다."""
        assert get_parser(Path("a.csv")).__class__ is CsvParser
        assert get_parser(Path("a.hwpx")).__class__ is HwpxParser

    def test_get_parser_returns_none_for_unknown(self) -> None:
        """미지원 확장자는 None 을 반환한다."""
        assert get_parser(Path("a.unknown")) is None


class TestLoaderDelegation:
    """DocumentLoader 가 레지스트리에 위임하는지 검증"""

    def test_loader_uses_registry(self, tmp_path: Path) -> None:
        """로더가 레지스트리 파서로 분기한다."""
        loader = DocumentLoader()
        csv_path = tmp_path / "x.csv"
        csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

        parser = loader.get_parser(csv_path)
        assert parser is not None
        assert parser.__class__ is CsvParser

    def test_load_file_parses_csv_end_to_end(self, tmp_path: Path) -> None:
        """load_file 이 csv 를 텍스트로 추출한다."""
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("이름,점수\n홍길동,90\n", encoding="utf-8")

        doc = load_file(csv_path)
        assert "이름 | 점수" in doc.content
        assert "홍길동 | 90" in doc.content


class TestCsvParser:
    """CSV 파서 테스트 (표준 라이브러리)"""

    def test_rows_joined_with_separator(self, tmp_path: Path) -> None:
        """각 행의 셀이 ' | ' 로 연결된다."""
        path = tmp_path / "a.csv"
        path.write_text("col1,col2\nv1,v2\n", encoding="utf-8")

        text = CsvParser().parse(path)
        assert text == "col1 | col2\nv1 | v2"

    def test_cp949_fallback(self, tmp_path: Path) -> None:
        """UTF-8 디코딩 실패 시 cp949 로 폴백한다."""
        path = tmp_path / "korean.csv"
        path.write_bytes("이름,부서\n홍길동,영업\n".encode("cp949"))

        text = CsvParser().parse(path)
        assert "홍길동 | 영업" in text


class TestHwpxParser:
    """HWPX 파서 테스트 (표준 라이브러리로 합성한 OWPML zip)"""

    def _make_hwpx(self, path: Path, body: str) -> None:
        """단일 문단을 담은 최소 HWPX(zip) 파일을 만든다."""
        section_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<hs:sec xmlns:hs="http://www.hancom.co.kr/hwpml/2011/section"'
            ' xmlns:hp="http://www.hancom.co.kr/hwpml/2011/paragraph">'
            f"<hp:p><hp:run><hp:t>{body}</hp:t></hp:run></hp:p>"
            "</hs:sec>"
        ).encode("utf-8")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("Contents/section0.xml", section_xml)

    def test_extracts_paragraph_text(self, tmp_path: Path) -> None:
        """section XML 의 문단 텍스트를 추출한다."""
        path = tmp_path / "doc.hwpx"
        self._make_hwpx(path, "한글 문서 본문입니다")

        text = HwpxParser().parse(path)
        assert text == "한글 문서 본문입니다"

    def test_invalid_zip_returns_empty(self, tmp_path: Path) -> None:
        """zip 이 아닌 파일은 빈 문자열을 반환한다(예외 전파 없음)."""
        path = tmp_path / "broken.hwpx"
        path.write_text("not a zip", encoding="utf-8")

        assert HwpxParser().parse(path) == ""


class TestDocxParser:
    """Word(.docx) 파서 테스트 (python-docx round-trip)"""

    def test_extracts_paragraphs_and_tables(self, tmp_path: Path) -> None:
        """문단과 표 텍스트를 추출한다."""
        docx = pytest.importorskip("docx")
        from rag.ingestion.parsers.docx import DocxParser

        path = tmp_path / "a.docx"
        document = docx.Document()
        document.add_paragraph("첫 문단입니다")
        table = document.add_table(rows=1, cols=2)
        table.rows[0].cells[0].text = "이름"
        table.rows[0].cells[1].text = "값"
        document.save(str(path))

        text = DocxParser().parse(path)
        assert "첫 문단입니다" in text
        assert "이름 | 값" in text


class TestXlsxParser:
    """Excel(.xlsx) 파서 테스트 (openpyxl round-trip)"""

    def test_extracts_sheet_rows(self, tmp_path: Path) -> None:
        """시트명 머리말과 행 텍스트를 추출한다."""
        openpyxl = pytest.importorskip("openpyxl")
        from rag.ingestion.parsers.xlsx import XlsxParser

        path = tmp_path / "a.xlsx"
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "데이터"
        sheet.append(["헤더A", "헤더B"])
        sheet.append(["값1", "값2"])
        workbook.save(str(path))

        text = XlsxParser().parse(path)
        assert "# 데이터" in text
        assert "헤더A | 헤더B" in text
        assert "값1 | 값2" in text


class TestPptxParser:
    """PowerPoint(.pptx) 파서 테스트 (python-pptx round-trip)"""

    def test_extracts_slide_text(self, tmp_path: Path) -> None:
        """슬라이드 머리말과 도형 텍스트를 추출한다."""
        pptx = pytest.importorskip("pptx")
        from rag.ingestion.parsers.pptx import PptxParser

        path = tmp_path / "a.pptx"
        presentation = pptx.Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[6])  # 빈 레이아웃
        textbox = slide.shapes.add_textbox(
            pptx.util.Inches(1), pptx.util.Inches(1), pptx.util.Inches(4), pptx.util.Inches(1)
        )
        textbox.text_frame.text = "슬라이드 텍스트"
        presentation.save(str(path))

        text = PptxParser().parse(path)
        assert "# 슬라이드 1" in text
        assert "슬라이드 텍스트" in text
