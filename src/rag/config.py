"""설정 관리 모듈

YAML 파일에서 설정을 로딩하고 Pydantic으로 검증합니다.
환경변수로 설정을 오버라이드할 수 있습니다 (RAG_* prefix).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# .env 파일 로드
load_dotenv()


class ProjectConfig(BaseModel):
    """프로젝트 기본 정보"""
    name: str = "terminal-rag"
    version: str = "0.1.0"
    index_path: str = "data/index"


class IngestionConfig(BaseModel):
    """문서 수집 설정"""
    supported_extensions: list[str] = Field(default=[".txt", ".md"])
    encoding: str = "utf-8"


class ChunkingConfig(BaseModel):
    """청킹 설정"""
    chunk_size: int = Field(default=1000, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0)
    preserve_structure: bool = True


class QdrantConfig(BaseModel):
    """Qdrant 서버 설정"""
    host: str = "localhost"
    port: int = 6333
    collection: str = "terminal-rag"


class EmbeddingConfig(BaseModel):
    """임베딩 설정"""
    # model: str = "text-embedding-3-small"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = Field(default=100, ge=1)
    store_type: Literal["faiss", "qdrant"] = "faiss"
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)


class RetrievalConfig(BaseModel):
    """검색 설정"""
    top_k: int = Field(default=5, ge=1, le=100)
    score_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    search_type: Literal["vector", "hybrid"] = "vector"
    # Reranker 설정
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


class OllamaConfig(BaseModel):
    """Ollama 설정"""
    # 기본값을 제거하여 환경변수 설정을 유도하거나, 명시적으로 로컬호스트임을 인지하고 사용하도록 함
    # 실운영 환경에서는 반드시 환경변수로 주입받아야 함
    base_url: str | None = Field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL"))
    model: str | None = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL"))


class GenerationConfig(BaseModel):
    """답변 생성 설정"""
    provider: Literal["gemini", "ollama"] = "gemini"
    model: str = "gemini-2.5-flash"  # 기본 모델 (provider에 따라 무시될 수 있음)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    
    # Provider-specific configs
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)


class LoggingConfig(BaseModel):
    """로깅 설정"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    file: str = "logs/rag.log"


class Config(BaseModel):
    """전체 설정"""
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환"""
    current = Path(__file__).resolve()
    # src/rag/config.py -> 프로젝트 루트
    return current.parent.parent.parent


def _load_yaml_config(config_path: Path) -> dict:
    """YAML 파일에서 설정 로딩"""
    if not config_path.exists():
        return {}
    
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config_dict: dict) -> dict:
    """환경변수로 설정 오버라이드 (RAG_* prefix)
    
    예: RAG_LOGGING_LEVEL=DEBUG -> logging.level = DEBUG
    """
    for key, value in os.environ.items():
        if not key.startswith("RAG_"):
            continue
        
        # RAG_LOGGING_LEVEL -> ["logging", "level"]
        parts = key[4:].lower().split("_")
        
        # 중첩 딕셔너리 탐색 및 설정
        current = config_dict
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # 마지막 키에 값 설정
        if parts:
            current[parts[-1]] = value
    
    return config_dict


def load_config(config_path: Path | str | None = None) -> Config:
    """설정 파일 로딩 및 검증
    
    Args:
        config_path: 설정 파일 경로. None이면 기본 경로 사용.
        
    Returns:
        검증된 Config 객체
    """
    if config_path is None:
        # 1. 환경변수 확인
        env_path = os.getenv("RAG_CONFIG_PATH")
        if env_path:
            config_path = Path(env_path)
        # 2. 현재 작업 디렉토리의 configs/default.yaml 확인 (Docker/배포 환경)
        elif (Path.cwd() / "configs" / "default.yaml").exists():
            config_path = Path.cwd() / "configs" / "default.yaml"
        # 3. 패키지 내부 경로 확인 (개발 환경)
        else:
            config_path = _get_project_root() / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)
    
    config_dict = _load_yaml_config(config_path)
    config_dict = _apply_env_overrides(config_dict)
    
    return Config.model_validate(config_dict)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """설정 싱글톤 반환 (캐싱됨)"""
    return load_config()


def reload_config() -> Config:
    """설정 캐시 초기화 후 다시 로딩"""
    get_config.cache_clear()
    return get_config()
