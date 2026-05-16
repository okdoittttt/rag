from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from pathlib import Path
from rag.generation.prompt import get_system_prompt

router = APIRouter()

# Docker에서는 /app/data, 로컬에서는 프로젝트 루트의 data/ 사용
def get_data_path() -> Path:
    docker_path = Path("/app/data")
    if docker_path.exists():
        return docker_path
    # 로컬 개발: 프로젝트 루트 기준 data/ 폴더
    return Path(__file__).parent.parent.parent.parent / "data"

class SystemPromptRequest(BaseModel):
    system_prompt: str

@router.get("/system-prompt")
async def get_system_prompt_endpoint():
    """현재 시스템 프롬프트를 조회합니다."""
    return {"system_prompt": get_system_prompt()}

@router.post("/system-prompt")
async def update_system_prompt(request: SystemPromptRequest):
    """시스템 프롬프트를 업데이트합니다."""
    try:
        data_path = get_data_path()
        prompt_path = data_path / "system_prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(request.system_prompt, encoding="utf-8")
        return {"message": "System prompt updated successfully", "system_prompt": request.system_prompt}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

