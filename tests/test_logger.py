"""로깅 모듈 테스트"""

import json
import logging
from io import StringIO

import structlog

from rag.logger import setup_logging, get_logger, _mask_sensitive_data


class TestLogger:
    """로거 테스트"""
    
    def test_setup_logging_runs(self) -> None:
        """로깅 설정이 에러 없이 실행되는지 확인"""
        setup_logging()
        
        # structlog가 정상적으로 설정되었는지 확인
        logger = get_logger("test")
        assert logger is not None
    
    def test_get_logger_returns_bound_logger(self) -> None:
        """get_logger가 로거를 반환하는지 확인"""
        setup_logging()
        logger = get_logger("test_module")
        
        # structlog는 LazyProxy를 반환하므로 기본 속성 확인
        assert hasattr(logger, "info")
        assert hasattr(logger, "debug")
        assert hasattr(logger, "warning")
        assert hasattr(logger, "error")
    
    def test_mask_sensitive_data(self) -> None:
        """민감정보 마스킹 확인"""
        event_dict = {
            "user": "john",
            "api_key": "secret123",
            "password": "mypassword",
            "message": "hello",
        }
        
        result = _mask_sensitive_data(None, "info", event_dict)
        
        assert result["user"] == "john"
        assert result["api_key"] == "***MASKED***"
        assert result["password"] == "***MASKED***"
        assert result["message"] == "hello"
    
    def test_mask_case_insensitive(self) -> None:
        """대소문자 구분 없이 마스킹되는지 확인"""
        event_dict = {
            "API_KEY": "secret",
            "Authorization": "bearer token",
        }
        
        result = _mask_sensitive_data(None, "info", event_dict)
        
        assert result["API_KEY"] == "***MASKED***"
        assert result["Authorization"] == "***MASKED***"
