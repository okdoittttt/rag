"""CLI 테스트 모듈"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from cli.main import app
from rag.chunking.chunk import Chunk


runner = CliRunner()


class TestCLI:
    """CLI 명령어 테스트"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock out core RAG components"""
        # Patching paths based on cli.commands imports
        with patch("cli.commands.index.load_documents") as mock_load, \
             patch("cli.commands.index.chunk_document") as mock_chunk, \
             patch("cli.commands.index.Embedder") as mock_embedder, \
             patch("cli.commands.index.VectorStore") as mock_store, \
             patch("cli.commands.index.HybridSearcher") as mock_indexer, \
             patch("cli.commands.ask.Embedder") as mock_ask_embedder, \
             patch("cli.commands.ask.VectorStore") as mock_ask_store, \
             patch("cli.commands.ask.HybridSearcher") as mock_searcher, \
             patch("cli.commands.ask.GeminiLLM") as mock_llm, \
             patch("cli.commands.search.Embedder") as mock_search_embedder, \
             patch("cli.commands.search.VectorStore") as mock_search_store, \
             patch("cli.commands.search.HybridSearcher") as mock_debug_searcher:
            
            # Setup mocks
            mock_load.return_value = ["mock_doc"]
            mock_chunk.return_value = [Chunk(content="mock content")]
            
            # Indexer mock
            indexer_instance = MagicMock()
            mock_indexer.return_value = indexer_instance
            
            # Searcher mock
            searcher_instance = MagicMock()
            searcher_instance.search.return_value = [(Chunk(content="found content"), 0.9)]
            mock_searcher.return_value = searcher_instance
            mock_debug_searcher.return_value = searcher_instance
            
            # LLM mock
            llm_instance = MagicMock()
            llm_instance.generate.return_value = "Mock Answer"
            mock_llm.return_value = llm_instance
            
            yield {
                "load": mock_load,
                "indexer": indexer_instance,
                "searcher": searcher_instance,
                "llm": llm_instance
            }

    def test_help(self):
        """도움말 출력 확인"""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Terminal RAG CLI" in result.stdout
        
    def test_index_command(self, mock_components, tmp_path):
        """인덱스 명령어 테스트"""
        # 임시 문서 경로
        doc_path = tmp_path / "docs"
        doc_path.mkdir()
        (doc_path / "test.txt").write_text("content")
        
        with patch("cli.commands.index.get_config") as mock_config:
            config_mock = MagicMock()
            config_mock.project.index_path = str(tmp_path / "rag_index")
            config_mock.embedding.dimension = 384
            mock_config.return_value = config_mock
            
            # 실행
            result = runner.invoke(app, ["index", str(doc_path)])
            
            if result.exit_code != 0:
                print(f"Stdout: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            assert "Successfully indexed" in result.stdout
        mock_components["load"].assert_called()
        mock_components["indexer"].index.assert_called()
        mock_components["indexer"].save.assert_called()

    def test_ask_command(self, mock_components, tmp_path):
        """Ask 명령어 테스트 (인덱스 필요)"""
        with patch("cli.commands.ask.get_config") as mock_config:
            # Config Mocking (dimension int 처리 필수)
            config_mock = MagicMock()
            config_mock.project.index_path = str(tmp_path / "index")
            config_mock.embedding.dimension = 384
            mock_config.return_value = config_mock
            
            (tmp_path / "index").mkdir()  # 인덱스 디렉토리 생성
            
            result = runner.invoke(app, ["ask", "test question"])
            
            if result.exit_code != 0:
                print(f"Stdout: {result.stdout}")
                print(f"Exception: {result.exception}")
            
            assert result.exit_code == 0
            assert "Answer:" in result.stdout
            assert "Mock Answer" in result.stdout
            assert "Answer:" in result.stdout
            assert "Mock Answer" in result.stdout
            # 검색 및 생성 호출 확인
            mock_components["searcher"].load.assert_called()
            mock_components["searcher"].search.assert_called()
            mock_components["llm"].generate.assert_called()

    def test_search_command(self, mock_components, tmp_path):
        """Search 명령어 테스트"""
        with patch("cli.commands.search.get_config") as mock_config:
            config_mock = MagicMock()
            config_mock.project.index_path = str(tmp_path / "index")
            config_mock.embedding.dimension = 384
            mock_config.return_value = config_mock
            
            (tmp_path / "index").mkdir()
            
            result = runner.invoke(app, ["search", "test query"])
            
            if result.exit_code != 0:
                print(f"Stdout: {result.stdout}")
                print(f"Exception: {result.exception}")
    
            assert result.exit_code == 0
            # 테이블 컬럼 확인
            assert "Rank" in result.stdout
            assert "found content" in result.stdout
