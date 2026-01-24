"""Semantic chunking module

Groups sentences based on semantic similarity using embeddings.
Maintains topic coherence while respecting size constraints.
"""

from __future__ import annotations

import re
import numpy as np
from typing import TYPE_CHECKING

from rag.chunking.chunk import Chunk
from rag.ingestion.language import detect_language
from rag.logger import get_logger

if TYPE_CHECKING:
    from rag.embedding.embedder import Embedder


logger = get_logger(__name__)

# Lazy import for Kiwi (optional dependency handling)
try:
    from kiwipiepy import Kiwi
    _kiwi_instance = None
    HAS_KIWI = True
except ImportError:
    HAS_KIWI = False
    logger.warning("kiwipiepy_not_available", fallback="regex_splitting")


# English sentence splitting pattern
# Matches: . ? ! followed by space and capital letter, or end of string
ENGLISH_SENTENCE_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$',
    re.MULTILINE
)

# Minimum text length for semantic chunking
MIN_TEXT_LENGTH = 50

# Maximum sentence length before sub-splitting
MAX_SENTENCE_LENGTH = 2000


def _get_kiwi() -> Kiwi:
    """Get lazy-initialized Kiwi instance"""
    global _kiwi_instance
    if _kiwi_instance is None:
        _kiwi_instance = Kiwi()
        logger.debug("kiwi_initialized")
    return _kiwi_instance


def _split_long_sentence(sentence: str, max_len: int = MAX_SENTENCE_LENGTH) -> list[str]:
    """Split overly long sentence at natural boundaries

    Args:
        sentence: Long sentence to split
        max_len: Maximum length per part

    Returns:
        List of sentence parts
    """
    if len(sentence) <= max_len:
        return [sentence]

    # Split on commas, semicolons, or whitespace
    parts = []
    current = ""

    # Try splitting on commas and semicolons first
    for segment in re.split(r'([,;])', sentence):
        if len(current) + len(segment) <= max_len:
            current += segment
        else:
            if current.strip():
                parts.append(current.strip())
            current = segment

    # If still too long, split on whitespace
    if current and len(current) > max_len:
        for word in current.split():
            if len(current) + len(word) + 1 <= max_len:
                current += (" " + word if current else word)
            else:
                if current:
                    parts.append(current)
                current = word

    if current.strip():
        parts.append(current.strip())

    return parts if parts else [sentence]


def _split_sentences_korean(text: str) -> list[str]:
    """Split Korean text into sentences using Kiwi

    Args:
        text: Korean text

    Returns:
        List of sentences
    """
    if not HAS_KIWI:
        logger.warning("kiwi_not_available", fallback="english_pattern")
        return _split_sentences_english(text)

    kiwi = _get_kiwi()
    sentences = []

    try:
        for sent in kiwi.split_into_sents(text):
            sentence_text = sent.text.strip()
            if sentence_text:
                # Handle very long sentences
                if len(sentence_text) > MAX_SENTENCE_LENGTH:
                    sentences.extend(_split_long_sentence(sentence_text))
                else:
                    sentences.append(sentence_text)
    except Exception as e:
        logger.error("kiwi_split_failed", error=str(e), fallback="regex")
        return _split_sentences_english(text)

    return sentences


def _split_sentences_english(text: str) -> list[str]:
    """Split English text into sentences using regex

    Args:
        text: English text

    Returns:
        List of sentences
    """
    # Simple sentence splitting on period, question mark, exclamation
    sentences = ENGLISH_SENTENCE_PATTERN.split(text)

    # Clean and filter
    result = []
    for sent in sentences:
        sent = sent.strip()
        if sent:
            # Handle very long sentences
            if len(sent) > MAX_SENTENCE_LENGTH:
                result.extend(_split_long_sentence(sent))
            else:
                result.append(sent)

    return result


def split_into_sentences(text: str, language: str | None = None) -> list[str]:
    """Split text into sentences based on language

    Args:
        text: Text to split
        language: Language code ('ko', 'en', or None for auto-detect)

    Returns:
        List of sentences
    """
    if not text or not text.strip():
        return []

    # Auto-detect language if not provided
    if language is None:
        language = detect_language(text)
        logger.debug("language_detected", language=language)

    # Choose appropriate splitter
    if language == "ko":
        return _split_sentences_korean(text)
    else:
        return _split_sentences_english(text)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors

    Note: Assumes vectors are already normalized.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score (0.0 to 1.0, typically)
    """
    return float(np.dot(vec1, vec2))


def _calculate_group_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """Calculate average embedding for a group of sentences

    Args:
        embeddings: List of sentence embeddings

    Returns:
        Average embedding (normalized)
    """
    if not embeddings:
        return np.array([])

    # Average and re-normalize
    avg = np.mean(embeddings, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm

    return avg


def _group_sentences_semantically(
    sentences: list[str],
    embeddings: np.ndarray,
    similarity_threshold: float,
    chunk_size: int,
) -> list[tuple[list[str], list[np.ndarray]]]:
    """Group sentences based on semantic similarity

    Args:
        sentences: List of sentences
        embeddings: Sentence embeddings (shape: [n_sentences, embedding_dim])
        similarity_threshold: Minimum similarity to stay in same group
        chunk_size: Maximum chunk size in characters

    Returns:
        List of (sentence_group, embedding_group) tuples
    """
    if not sentences:
        return []

    groups = []
    current_sentences = []
    current_embeddings = []
    current_size = 0

    for i, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        sentence_len = len(sentence)

        # Check if we should add to current group
        should_add = False
        similarity = 0.0

        if not current_embeddings:
            # First sentence always starts a group
            should_add = True
        else:
            # Calculate similarity with current group
            group_embedding = _calculate_group_embedding(current_embeddings)
            similarity = cosine_similarity(group_embedding, embedding)

            # Add if similar AND doesn't exceed size
            if similarity >= similarity_threshold:
                # +1 for space between sentences
                if current_size + sentence_len + 1 <= chunk_size:
                    should_add = True
                else:
                    # Similar but too large - need to split
                    logger.debug(
                        "semantic_group_size_limit",
                        similarity=round(similarity, 3),
                        current_size=current_size,
                        sentence_len=sentence_len,
                    )

        if should_add:
            current_sentences.append(sentence)
            current_embeddings.append(embedding)
            current_size += sentence_len + (1 if len(current_sentences) > 1 else 0)
        else:
            # Start new group
            if current_sentences:
                groups.append((current_sentences, current_embeddings))
                logger.debug(
                    "semantic_group_created",
                    sentences=len(current_sentences),
                    size=current_size,
                    reason="similarity_drop" if similarity < similarity_threshold else "size_limit",
                )

            current_sentences = [sentence]
            current_embeddings = [embedding]
            current_size = sentence_len

    # Add final group
    if current_sentences:
        groups.append((current_sentences, current_embeddings))
        logger.debug(
            "semantic_group_created",
            sentences=len(current_sentences),
            size=current_size,
            reason="last_group",
        )

    return groups


def _apply_overlap_to_chunks(
    chunks: list[Chunk],
    chunk_overlap: int,
    chunk_size: int,
) -> list[Chunk]:
    """Apply overlap between chunks by prepending sentences from previous chunk

    Args:
        chunks: List of chunks without overlap
        chunk_overlap: Target overlap size in characters
        chunk_size: Maximum chunk size (to prevent exceeding limit)

    Returns:
        Chunks with overlap applied
    """
    if not chunks or chunk_overlap <= 0:
        return chunks

    overlapped = [chunks[0]]  # First chunk unchanged

    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        current_chunk = chunks[i]

        # Get sentences from previous chunk for overlap
        prev_sentences = split_into_sentences(prev_chunk.content)
        overlap_text = ""

        # Add sentences from end of previous chunk until we reach overlap size
        # But ensure total size doesn't exceed chunk_size
        for sent in reversed(prev_sentences):
            potential_size = len(overlap_text) + len(sent) + 1 + len(current_chunk.content)
            if (len(overlap_text) + len(sent) + 1 <= chunk_overlap and
                potential_size <= chunk_size):
                overlap_text = sent + (" " + overlap_text if overlap_text else "")
            else:
                break

        # Prepend overlap to current chunk
        if overlap_text.strip():
            new_content = overlap_text.strip() + " " + current_chunk.content
            # Update chunk content only if it doesn't exceed size limit
            if len(new_content) <= chunk_size:
                current_chunk.content = new_content

        overlapped.append(current_chunk)

    return overlapped


def split_semantic(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    similarity_threshold: float = 0.7,
    source: str = "",
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Split text into semantically coherent chunks

    Args:
        text: Text to split
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap size in characters
        similarity_threshold: Minimum cosine similarity to group sentences (0.0-1.0)
        source: Source document path (for metadata)
        embedder: Embedder instance (will create if None)

    Returns:
        List of Chunk objects
    """
    if not text or not text.strip():
        logger.debug("empty_text_provided")
        return []

    # Fallback for very short text
    if len(text) < MIN_TEXT_LENGTH:
        logger.info(
            "text_too_short_for_semantic",
            length=len(text),
            using="simple_chunking",
        )
        # Import here to avoid circular dependency
        from rag.chunking.splitter import split_text
        return split_text(text, chunk_size, chunk_overlap, source)

    # Initialize embedder if needed
    if embedder is None:
        from rag.embedding.embedder import Embedder
        embedder = Embedder()

    # Step 1: Split into sentences
    logger.debug("splitting_sentences", text_length=len(text))
    try:
        sentences = split_into_sentences(text)
    except Exception as e:
        logger.error("sentence_split_failed", error=str(e), using="simple_chunking")
        from rag.chunking.splitter import split_text
        return split_text(text, chunk_size, chunk_overlap, source)

    if not sentences:
        logger.warning("no_sentences_found", text_preview=text[:100])
        return []

    # Single sentence document - return as single chunk
    if len(sentences) == 1:
        logger.debug("single_sentence_document")
        chunk = Chunk.create(
            content=text.strip(),
            source=source,
            chunk_index=0,
            start_char=0,
            end_char=len(text),
        )
        chunk.metadata["chunking_strategy"] = "semantic"
        chunk.metadata["sentence_count"] = 1
        chunk.metadata["avg_similarity"] = 1.0
        return [chunk]

    logger.info("sentences_split", count=len(sentences))

    # Step 2: Embed all sentences
    logger.debug("embedding_sentences", count=len(sentences))
    try:
        embeddings = embedder.embed(sentences)
    except Exception as e:
        logger.error("embedding_failed", error=str(e), using="simple_chunking")
        from rag.chunking.splitter import split_text
        return split_text(text, chunk_size, chunk_overlap, source)

    # Step 3: Group sentences semantically
    logger.debug(
        "grouping_semantically",
        threshold=similarity_threshold,
        chunk_size=chunk_size,
    )
    groups = _group_sentences_semantically(
        sentences=sentences,
        embeddings=embeddings,
        similarity_threshold=similarity_threshold,
        chunk_size=chunk_size,
    )

    logger.info("semantic_groups_created", count=len(groups))

    # Step 4: Create chunks from groups
    chunks = []
    char_position = 0

    for chunk_idx, (sent_group, embed_group) in enumerate(groups):
        # Combine sentences
        content = " ".join(sent_group)
        chunk_len = len(content)

        # Calculate average similarity within group (for metadata)
        avg_similarity = 0.0
        if len(embed_group) > 1:
            group_emb = _calculate_group_embedding(embed_group)
            similarities = [
                cosine_similarity(group_emb, emb) for emb in embed_group
            ]
            avg_similarity = float(np.mean(similarities))

        chunk = Chunk.create(
            content=content,
            source=source,
            chunk_index=chunk_idx,
            start_char=char_position,
            end_char=char_position + chunk_len,
        )

        # Add semantic-specific metadata
        chunk.metadata["chunking_strategy"] = "semantic"
        chunk.metadata["sentence_count"] = len(sent_group)
        chunk.metadata["avg_similarity"] = round(avg_similarity, 3)

        chunks.append(chunk)
        char_position += chunk_len + 1  # +1 for space between chunks

    # Step 5: Apply overlap
    if chunk_overlap > 0 and len(chunks) > 1:
        logger.debug("applying_overlap", overlap=chunk_overlap)
        chunks = _apply_overlap_to_chunks(chunks, chunk_overlap, chunk_size)

    avg_chunk_size = int(np.mean([len(c.content) for c in chunks])) if chunks else 0
    logger.info(
        "semantic_chunking_complete",
        chunks=len(chunks),
        avg_chunk_size=avg_chunk_size,
    )

    return chunks
