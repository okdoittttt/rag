"""RAG 평가 파이프라인 패키지.

골든셋 기반 평가, 커스텀 메트릭(검색 단계), RAGAS 메트릭(LLM-as-judge),
리포트 생성을 한 곳에 모은다. 본 패키지는 ``optional-dependencies`` 의
``eval`` 그룹에만 의존하며 운영 코드 경로에는 포함되지 않는다.

본 프로젝트는 src-layout(``src/rag``, ``src/api``)을 채택하면서도
``[build-system]`` 을 두지 않아 ``uv run`` 환경에서는 ``src`` 가
``sys.path`` 에 자동 추가되지 않는다(``[tool.pytest.ini_options].pythonpath``
는 pytest 전용). 평가 CLI를 ``uv run python -m evals.ragas_runner`` 로
직접 호출할 수 있도록, 패키지 로드 시점에 한 번 경로를 보정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
