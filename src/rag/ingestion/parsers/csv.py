"""CSV(.csv) 파서

표준 라이브러리 ``csv`` 모듈로 .csv 파일을 행 단위 텍스트로 변환합니다.
"""

from __future__ import annotations

import csv
from pathlib import Path

from rag.ingestion.parsers.base import DocumentParser
from rag.logger import get_logger


logger = get_logger(__name__)


class CsvParser(DocumentParser):
    """CSV(.csv) 파일 파서"""

    extensions = [".csv"]

    def parse(self, path: Path, encoding: str = "utf-8") -> str:
        """.csv 파일에서 텍스트를 추출한다.

        각 행의 비어있지 않은 셀을 ``" | "``로 연결한다. UTF-8 디코딩 실패 시
        ``cp949`` 로 재시도하여 한국어 CSV 호환성을 확보한다. ``import csv`` 는
        파이썬 절대 import 규칙에 따라 표준 라이브러리를 가리키므로 본 모듈명과
        충돌하지 않는다.

        Args:
            path: 파싱할 .csv 파일 경로.
            encoding: 1차 시도 인코딩. 기본값은 ``"utf-8"``.

        Returns:
            추출된 텍스트. 추출 실패 시 빈 문자열.
        """
        for enc in (encoding, "cp949"):
            try:
                with path.open("r", encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    lines: list[str] = []
                    for row in reader:
                        cells = [cell.strip() for cell in row if cell.strip()]
                        if cells:
                            lines.append(" | ".join(cells))
                    return "\n".join(lines)
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning("csv_parse_failed", path=str(path), error=str(e))
                return ""

        logger.warning("csv_decode_failed", path=str(path))
        return ""
