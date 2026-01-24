"""의미론적 청킹 사용 예제"""

from rag.chunking import chunk_document
from rag.ingestion.document import Document
from rag.config import get_config


def main():
    # 테스트 텍스트 (서로 다른 주제)
    test_text = """
    Python is a high-level programming language. It was created by Guido van Rossum.
    Python emphasizes code readability and simplicity. Many developers love Python for its clean syntax.

    Machine learning is a subset of artificial intelligence. It focuses on training algorithms to learn from data.
    Neural networks are a key component of machine learning. Deep learning uses multiple layers of neural networks.

    The weather today is sunny and warm. It's a perfect day for outdoor activities.
    Many people enjoy going to the park on days like this. The temperature is around 25 degrees Celsius.
    """

    print("=== 의미론적 청킹 테스트 ===\n")

    # 1. 기본 텍스트 전략으로 청킹
    print("1. 기본 텍스트 전략 (strategy='text'):")
    config = get_config()
    original_strategy = config.chunking.strategy
    config.chunking.strategy = "text"

    doc = Document(content=test_text, metadata={"extension": ".txt", "source": "test.txt"})
    chunks_text = chunk_document(doc)

    print(f"   청크 수: {len(chunks_text)}")
    for i, chunk in enumerate(chunks_text):
        print(f"   청크 {i}: {len(chunk.content)}자")
        print(f"   내용: {chunk.content[:80]}...")
        print()

    # 2. 의미론적 전략으로 청킹
    print("\n2. 의미론적 전략 (strategy='semantic', threshold=0.7):")
    config.chunking.strategy = "semantic"
    config.chunking.semantic_threshold = 0.7

    chunks_semantic = chunk_document(doc)

    print(f"   청크 수: {len(chunks_semantic)}")
    for i, chunk in enumerate(chunks_semantic):
        strategy = chunk.metadata.get("chunking_strategy", "unknown")
        sent_count = chunk.metadata.get("sentence_count", "?")
        avg_sim = chunk.metadata.get("avg_similarity", "?")

        print(f"   청크 {i}:")
        print(f"     - 크기: {len(chunk.content)}자")
        print(f"     - 전략: {strategy}")
        print(f"     - 문장 수: {sent_count}")
        print(f"     - 평균 유사도: {avg_sim}")
        print(f"     - 내용: {chunk.content[:100]}...")
        print()

    # 3. 낮은 threshold로 더 세밀하게 분할
    print("\n3. 낮은 threshold로 세밀한 분할 (threshold=0.5):")
    config.chunking.semantic_threshold = 0.5

    chunks_low_threshold = chunk_document(doc)

    print(f"   청크 수: {len(chunks_low_threshold)}")
    for i, chunk in enumerate(chunks_low_threshold):
        sent_count = chunk.metadata.get("sentence_count", "?")
        print(f"   청크 {i}: {len(chunk.content)}자, {sent_count}개 문장")

    # 설정 복원
    config.chunking.strategy = original_strategy

    print("\n=== 테스트 완료 ===")
    print(f"\n요약:")
    print(f"  - 기본 텍스트 전략: {len(chunks_text)}개 청크")
    print(f"  - 의미론적 전략 (0.7): {len(chunks_semantic)}개 청크")
    print(f"  - 의미론적 전략 (0.5): {len(chunks_low_threshold)}개 청크")
    print(f"\n의미론적 청킹은 주제 전환을 감지하여 더 의미있는 단위로 분할합니다.")


if __name__ == "__main__":
    main()
