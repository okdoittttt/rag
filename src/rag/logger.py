"""구조화 로깅 시스템

structlog 기반의 JSON 포맷 로깅을 제공합니다.
민감정보는 자동으로 마스킹됩니다.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from rag.config import get_config


# 마스킹할 키워드 목록
SENSITIVE_KEYS = {
    "password", "token", "api_key", "apikey", "secret", 
    "authorization", "auth", "credential", "private_key"
}


def _mask_sensitive_data(
    logger: structlog.types.WrappedLogger,  # noqa: ARG001
    method_name: str,  # noqa: ARG001
    event_dict: dict,
) -> dict:
    """민감정보 마스킹 프로세서"""
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(s in key_lower for s in SENSITIVE_KEYS):
            event_dict[key] = "***MASKED***"
    return event_dict


def _ensure_log_directory(log_path: str) -> Path:
    """로그 디렉토리가 없으면 생성"""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging() -> None:
    """로깅 시스템 초기화"""
    config = get_config()
    log_config = config.logging
    
    # 로그 레벨 설정
    log_level = getattr(logging, log_config.level.upper(), logging.INFO)
    
    # 출력 포맷 결정
    if log_config.format == "json":
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)
    
    # structlog 프로세서 체인
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _mask_sensitive_data,
        renderer,
    ]
    
    # structlog 설정
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # 표준 logging 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 제거
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # 파일 핸들러 (옵션)
    if log_config.file:
        log_path = _ensure_log_directory(log_config.file)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """로거 인스턴스 반환
    
    Args:
        name: 로거 이름. None이면 호출자 모듈명 사용.
        
    Returns:
        structlog BoundLogger 인스턴스
    """
    return structlog.get_logger(name)
