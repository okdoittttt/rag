"""기본 텍스트 분할 모듈

문자 수 기반으로 텍스트를 청크로 분할합니다.
문장/문단 경계를 최대한 보존하며, 강제 절단 시에도 단어 중간이 잘리지 않도록
단어 경계로 스냅합니다.
"""

from __future__ import annotations

import re

from rag.chunking.chunk import Chunk


# 분할 우선순위별 구분자 패턴
SEPARATORS = [
    "\n\n",    # 문단 경계
    "\n",      # 줄바꿈
    ". ",      # 문장 경계 (마침표)
    "? ",      # 문장 경계 (물음표)
    "! ",      # 문장 경계 (느낌표)
    "。",      # 한국어/일본어 마침표
    " ",       # 단어 경계
]

# 단어 경계로 인정하는 문자(공백류, 종결/구분 부호, CJK 종결 부호)
_WORD_BOUNDARY = re.compile(r"""[\s.?!,;:)\]}"'。、]""")

# 이 길이 미만의 청크는 단독으로 남기지 않고 인접 청크에 병합한다.
MIN_CHUNK_CONTENT_CHARS = 50


def _snap_to_word_boundary(text: str, max_size: int) -> int:
    """``max_size`` 인근의 가장 가까운 단어 경계로 분할 지점을 스냅한다.

    ``max_size`` 기준 ±윈도우 범위에서 직전/직후의 단어 경계를 찾아 ``max_size``에
    더 가까운 쪽을 선택한다. 윈도우 내에 경계가 전혀 없으면(긴 식별자/URL 등)
    ``max_size``로 폴백한다.

    Args:
        text: 분할할 텍스트.
        max_size: 목표 분할 크기(자).

    Returns:
        단어 경계 직후를 가리키는 분할 지점 인덱스.
    """
    window = max(int(max_size * 0.1), 16)
    lo = max(1, max_size - window)
    hi = min(len(text), max_size + window)

    before: int | None = None  # max_size 이하의 가장 가까운 경계 분할점
    after: int | None = None   # max_size 초과의 가장 가까운 경계 분할점

    # 경계 문자 위치 i에서의 분할점은 i+1(경계 직후)이다.
    for i in range(lo - 1, hi):
        if _WORD_BOUNDARY.match(text[i]):
            split_point = i + 1
            if split_point <= max_size:
                before = split_point
            else:
                after = split_point
                break

    if before is not None and after is not None:
        return before if (max_size - before) <= (after - max_size) else after
    if before is not None:
        return before
    if after is not None:
        return after
    return max_size


def _find_split_point(text: str, max_size: int) -> int:
    """최적의 분할 지점을 찾는다.

    우선순위에 따라 구분자를 탐색해 분할 지점을 결정한다. 구분자를 찾지 못하면
    ``max_size``에서 문자를 강제 절단하는 대신, ``max_size`` 인근의 가장 가까운
    단어 경계로 스냅하여 단어 중간이 잘리는 것을 방지한다.

    Args:
        text: 분할할 텍스트.
        max_size: 최대 청크 크기(자).

    Returns:
        분할 지점 인덱스(경계 직후 위치).
    """
    if len(text) <= max_size:
        return len(text)

    # 각 구분자에 대해 max_size 이하에서 가장 마지막 위치 찾기
    for sep in SEPARATORS:
        # max_size 범위 내에서 구분자의 마지막 위치 찾기
        search_area = text[:max_size]
        pos = search_area.rfind(sep)

        if pos > 0:
            # 구분자 포함하여 분할 (문장 부호는 앞 청크에 포함)
            return pos + len(sep)

    # 구분자를 찾지 못하면 단어 경계로 스냅 (실패 시에만 max_size로 폴백)
    return _snap_to_word_boundary(text, max_size)


def _snap_overlap_start(text: str, proposed: int, limit: int) -> int:
    """오버랩 시작점을 직후 첫 단어 경계 이후로 이동시킨다.

    오버랩으로 계산된 잠정 시작 위치가 단어 중간이 되지 않도록, 해당 위치 이후의
    첫 단어 경계 다음으로 이동한다. ``limit``(현재 청크 끝)을 넘어 이동하지 않으므로
    원문 커버리지에 공백이 생기지 않는다.

    Args:
        text: 원본 텍스트.
        proposed: 오버랩으로 계산된 잠정 시작 위치.
        limit: 탐색 상한(현재 청크 끝 위치).

    Returns:
        단어 경계 직후로 스냅된 시작 위치. 경계가 없으면 ``proposed`` 그대로.
    """
    if proposed <= 0:
        return proposed

    end = min(limit, len(text))
    for i in range(proposed, end):
        if _WORD_BOUNDARY.match(text[i]):
            return i + 1
    return proposed


def _merge_mini_chunks(
    raw: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    """과소 청크를 인접 청크에 병합한다.

    ``MIN_CHUNK_CONTENT_CHARS`` 미만의 청크는 단독으로 남기지 않고 다음 청크 앞에
    prepend하여 병합한다. 마지막 청크가 과소면 직전 청크에 합친다. PDF 헤더/풋터
    잔여물 같은 노이즈 청크를 줄이는 것이 목적이다. 병합 시 ``start_char``/
    ``end_char``는 두 청크를 아우르는 원문 범위로 갱신한다.

    Args:
        raw: ``(content, start_char, end_char)`` 튜플 리스트.

    Returns:
        병합 후 ``(content, start_char, end_char)`` 튜플 리스트.
    """
    if not raw:
        return []

    result: list[tuple[str, int, int]] = []
    pending: tuple[str, int, int] | None = None

    for content, sc, ec in raw:
        # 직전까지 누적된 과소 청크가 있으면 현재 청크 앞에 병합
        if pending is not None:
            content = pending[0] + " " + content
            sc = pending[1]
            ec = max(pending[2], ec)
            pending = None

        if len(content) < MIN_CHUNK_CONTENT_CHARS:
            pending = (content, sc, ec)
        else:
            result.append((content, sc, ec))

    # 마지막 청크가 과소로 남았으면 직전 청크에 합침
    if pending is not None:
        if result:
            pc, psc, pec = result[-1]
            result[-1] = (pc + " " + pending[0], psc, max(pec, pending[2]))
        else:
            # 전체가 하나의 과소 청크인 경우(짧은 텍스트) 그대로 유지
            result.append(pending)

    return result


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    source: str = "",
) -> list[Chunk]:
    """텍스트를 청크로 분할한다.

    구분자/단어 경계를 보존하며 텍스트를 분할하고, 오버랩 시작점도 단어 경계로
    스냅한다. 분할 후 과소 청크는 인접 청크에 병합하며, ``chunk_index``는 0부터
    빈틈없이 재부여한다.

    Args:
        text: 분할할 텍스트.
        chunk_size: 목표 청크 크기(자). 기본값 1000.
        chunk_overlap: 오버랩 크기(자). 기본값 150.
        source: 원본 문서 경로(메타데이터용).

    Returns:
        ``chunk_index`` 순서대로 정렬된 ``Chunk`` 리스트. 빈 텍스트는 ``[]``.
    """
    if not text or not text.strip():
        return []

    # 1단계: 원시 청크 추출 ((content, start_char, end_char))
    raw: list[tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        # 남은 텍스트
        remaining = text[start:]

        # 분할 지점 찾기
        split_point = _find_split_point(remaining, chunk_size)

        # 청크 내용 추출
        chunk_content = remaining[:split_point].strip()

        if chunk_content:  # 빈 청크 제외
            raw.append((chunk_content, start, start + split_point))

        # 다음 시작점 (오버랩 적용)
        if start + split_point >= len(text):
            break

        # 오버랩을 적용하되 단어 경계로 스냅, 최소한 1자는 진행
        proposed = start + split_point - chunk_overlap
        next_start = _snap_overlap_start(text, proposed, start + split_point)
        start = max(next_start, start + 1)

    # 2단계: 과소 청크 병합
    merged = _merge_mini_chunks(raw)

    # 3단계: Chunk 생성 (chunk_index 연속 부여)
    chunks: list[Chunk] = []
    for index, (content, start_char, end_char) in enumerate(merged):
        chunks.append(
            Chunk.create(
                content=content,
                source=source,
                chunk_index=index,
                start_char=start_char,
                end_char=end_char,
            )
        )

    return chunks
