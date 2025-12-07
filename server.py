import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent_core import run_agent_task
from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EvoAPI")

app = FastAPI(title="Evo Studio API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
# 出力ディレクトリがない場合の対策
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
# 出力ディレクトリをプレビュー用にマウント
app.mount("/preview", StaticFiles(directory=config.OUTPUT_DIR), name="preview")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
async def index():
    if os.path.exists("templates/index.html"):
        return FileResponse("templates/index.html")
    return {"message": "Welcome to Evo API. Please create templates/index.html"}

@app.post("/generate")
async def generate(req: PromptRequest):
    result = run_agent_task(req.prompt)
    
    # 成功フラグがFalseでも、成果物(files)がある場合は「部分的成功」として返す
    if not result["success"]:
        if result.get("files"):
            # エラーはあるがファイルは生成された場合
            result["success"] = True
            result["warning"] = result.get("error")
            del result["error"]
        else:
            return JSONResponse(content=result, status_code=200)
            
    return result

@app.get("/files")
async def list_files():
    files = []
    IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules", "venv", "_trash"}
    # 隠しファイルや不要な拡張子を除外
    IGNORE_EXTS = {".pyc", ".pyo", ".pyd", ".DS_Store", ".db", ".sqlite", ".png", ".jpg", ".jpeg", ".ico"}

    if not os.path.exists(config.OUTPUT_DIR):
        return {"files": []}

    for root, dirs, filenames in os.walk(config.OUTPUT_DIR):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext in IGNORE_EXTS: continue
                
            rel_path = os.path.relpath(os.path.join(root, filename), config.OUTPUT_DIR)
            files.append(rel_path.replace("\\", "/"))
            
    return {"files": files}

@app.get("/files/content")
async def get_file_content(filename: str):
    path = os.path.join(config.OUTPUT_DIR, filename)
    
    # パス・トラバーサル対策
    if not os.path.abspath(path).startswith(os.path.abspath(config.OUTPUT_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")
        
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(path, 'r', encoding='utf-8') as f:
            return {"content": f.read()}
    except UnicodeDecodeError:
        return {"content": "(Binary file)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)