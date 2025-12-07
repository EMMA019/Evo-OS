# Project Code Summary

Generated on: 2025-12-07 12:39:56

## File: `Dockerfile`

dockerfile
# ファイル名: Dockerfile
# 役割: Evo OS Core がコードを実行するための安全なサンドボックス環境定義
# ビルドコマンド: docker build -t evo-sandbox .

# 軽量かつ安定したPython環境をベースにする
FROM python:3.10-slim

# システムパッケージのインストール
# build-essential: C拡張のコンパイルに必要 (numpyなど)
# git, curl: 一般的なツール
# libxml2-dev, libxslt-dev: lxmlなどのパースライブラリ用
# nodejs, npm: React/Frontendのビルド用
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libxml2-dev \
    libxslt-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ワークスペースの設定（agent_core.pyのマウント先）
WORKDIR /workspace

# よく使われるPythonライブラリをプリインストール
# これにより、AIが生成したコードの ModuleNotFoundError を防ぎ、実行速度を上げる
# Qiskitなどの重いライブラリも含めることで「地獄級」タスクにも対応
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    flask \
    requests \
    beautifulsoup4 \
    lxml \
    matplotlib \
    pytest \
    scipy \
    scikit-learn \
    qiskit \
    fastapi \
    uvicorn \
    websockets

# コンテナが勝手に終了しないようにする（agent_core.pyが exec で入るため）
CMD ["sleep", "infinity"]


## File: `agent_core.py`

py
import os
import sys
import time
import json
import logging
import subprocess
import atexit
import uuid
import shutil
import contextvars
import re
from typing import Dict, List, Optional

# 設定とサービス群のインポート
from src.config import config
from src.services.budget_service import BudgetGuard
from src.services.workspace_manager import WorkspaceManager
from src.services.architect_service import ArchitectService
from src.services.kit_manager import KitManager
from src.services.kit_gen_service import KitGenService
from src.services.search_service import SearchService
from src.services.qa_service import QualityAssuranceService
from src.services.verifier_service import VerifierService
from src.services.healer_service import HealerService
from src.services.structure_service import StructureService
from src.services.data_recorder import DataRecorder

# ランタイムクラスの定義（簡略化のためここに配置。別ファイル分離を推奨）
class BaseRuntime:
    def start(self): pass
    def stop(self): pass
    def install_requirements(self): pass
    def test_run(self, entry_point): return True, "No runtime"
class DockerRuntime(BaseRuntime):
    def __init__(self):
        self.container = f"{config.CONTAINER_PREFIX}-{uuid.uuid4().hex[:8]}"
        self.workdir = os.path.abspath(config.OUTPUT_DIR)
        self._started = False
        self._available = bool(shutil.which("docker"))
        if self._available:
            try: subprocess.run(["docker", "info"], capture_output=True, check=True)
            except: self._available = False
    def start(self):
        if not self._available or self._started: return
        self._cleanup()
        try:
            env_args = ["-e", f"GOOGLE_API_KEY={config.LLM_API_KEY}"]
            subprocess.run(
                ["docker", "run", "-d", "--rm", "--name", self.container, "--network", "host", "-v", f"{self.workdir}:/workspace"] + env_args + [config.DOCKER_IMAGE, "sleep", "infinity"], 
                check=True, capture_output=True
            )
            self._started = True; atexit.register(self.stop)
            logger.info("🐳 Docker Runtime Started.")
        except Exception as e: 
            logger.warning(f"⚠️ Docker failed: {e}. Falling back to Local.")
            self._available = False
    def stop(self):
        if self._started: 
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True)
            self._started = False
            logger.info("🐳 Docker Runtime Stopped.")
    def _cleanup(self): self.stop()
    def install_requirements(self):
        if not self._started: return
        if os.path.exists(os.path.join(self.workdir, "requirements.txt")):
            logger.info("📦 Docker: Installing requirements...")
            subprocess.run(["docker", "exec", "-w", "/workspace", self.container, "pip", "install", "-r", "requirements.txt"], capture_output=True, timeout=120)
    def test_run(self, entry_point="app.py"):
        if not self._started: return False, "Docker not started"
        try:
            cmd = ["docker", "exec", "-w", "/workspace", self.container, "python", entry_point]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try: outs, errs = proc.communicate(timeout=10) 
            except subprocess.TimeoutExpired: proc.kill(); return True, "Running"
            if proc.returncode != 0: return False, f"Error:\n{errs}"
            return True, "Success"
        except Exception as e: return False, str(e)
class LocalRuntime(BaseRuntime):
    def __init__(self):
        self.workdir = os.path.abspath(config.OUTPUT_DIR)
        self.venv_dir = os.path.join(self.workdir, ".venv")
        is_win = os.name == 'nt'
        self.py_exe = os.path.join(self.venv_dir, "Scripts" if is_win else "bin", "python.exe" if is_win else "python")
    def start(self):
        if not os.path.exists(self.py_exe):
            logger.info("🐍 Creating Local venv...")
            subprocess.run([sys.executable, "-m", "venv", self.venv_dir], check=True)
        logger.info("🐍 Local Runtime Ready.")
    def install_requirements(self):
        req = os.path.join(self.workdir, "requirements.txt")
        if os.path.exists(req):
            logger.info("📦 Local: Installing requirements...")
            try: subprocess.run([self.py_exe, "-m", "pip", "install", "-r", req], cwd=self.workdir, capture_output=True, check=True, timeout=120)
            except: pass
    def test_run(self, entry_point="app.py"):
        if not os.path.exists(os.path.join(self.workdir, entry_point)): return False, "File not found"
        logger.info(f"🧪 Local Testing: {entry_point}...")
        try:
            env = os.environ.copy()
            env["GOOGLE_API_KEY"] = config.LLM_API_KEY
            proc = subprocess.Popen([self.py_exe, entry_point], cwd=self.workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try: outs, errs = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired: proc.kill(); return True, "Running"
            if proc.returncode != 0: return False, f"Error:\n{errs}\n{outs}"
            return True, "Success"
        except Exception as e: return False, str(e)


# --- ロギング設定 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("EvoCore")

# --- ヘルパー: パスの安全性確保 ---
def safe_path_join(base, *paths):
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)): raise ValueError("Path traversal attempt")
    return final_path

# --- AIクライアント設定 ---
try:
    import google.generativeai as genai
    if config.LLM_API_KEY: genai.configure(api_key=config.LLM_API_KEY)
except: pass

class ResilientClient:
    """LLM呼び出しクライアント: configのMAX_RETRIESに依存"""
    def __init__(self, model, budget_guard):
        self.model = genai.GenerativeModel(model)
        self.budget = budget_guard
        self.name = model

    def generate(self, prompt, sys_prompt="") -> str:
        full_prompt = f"{sys_prompt}\n\n{prompt}"
        
        # configのMAX_RETRIESを使用 (現在は1)
        for i in range(config.MAX_RETRIES):
            try:
                res = self.model.generate_content(full_prompt)
                text = res.text.strip()
                self.budget.check_and_record(self.name, len(full_prompt), len(text))
                return text
            except Exception as e:
                if "Budget" in str(e): raise e
                logger.warning(f"⚠️ GenAI Error ({i+1}/{config.MAX_RETRIES}): {e}")
                time.sleep(1)
        raise RuntimeError("LLM Error: Failed after all retries.")


class Orchestrator:
    """
    エージェントの司令塔（God Objectの分離完了）。
    各専門サービスを呼び出すことに徹する。
    """
    def __init__(self):
        self.logs = []
        self.budget = BudgetGuard(config.MAX_BUDGET_PER_RUN)
        
        # 1. Workspace & Git (雑務係)
        self.ws = WorkspaceManager()
        
        # 2. AI Clients (全モデルをStandard Flashに統一済み)
        client_fast = ResilientClient(config.LLM_MODEL_FAST, self.budget)
        client_smart = ResilientClient(config.LLM_MODEL_SMART, self.budget)
        client_healer = ResilientClient(config.LLM_MODEL_HEALER, self.budget)
        
        # 3. Services (依存関係の注入)
        self.kit_mgr = KitManager(client_fast)
        # ArchitectはKitManagerに依存する
        self.architect = ArchitectService(client_smart, self.kit_mgr)
        
        self.verifier = VerifierService(None) # Runtimeは後で注入
        self.healer = HealerService(client_fast, client_healer)
        self.qa = QualityAssuranceService(client_smart)
        self.structure = StructureService()
        self.search = SearchService(client_fast)
        self.kit_gen = KitGenService(client_smart)
        self.recorder = DataRecorder()

        # 4. Runtime
        self.docker = DockerRuntime()
        self.runtime = self.docker if self.docker._available else LocalRuntime()
        self.runtime.start()
        self.verifier.runtime = self.runtime # VerifierにRuntimeを注入

        # 実行中のKit情報を保持
        self.current_kit = None 

    def log(self, msg):
        logger.info(msg)
        self.logs.append(msg)

    def cleanup(self):
        self.runtime.stop()

    def run(self, prompt: str) -> Dict:
        """メイン実行フロー: 直列的で読みやすい構造"""
        self.log(f"🚀 Evo Started: {prompt[:30]}...")
        
        try:
            # A. 特殊モード判定
            if any(k in prompt for k in ["キットを作って", "Kitを作って", "Create Kit"]):
                return self._mode_kit_gen(prompt)
            if any(k in prompt.lower() for k in ["調べて", "search", "research"]):
                return self._mode_research(prompt)

            # B. 準備フェーズ: 計画作成とキット選択を一度に行う
            plan, kit = self.architect.create_plan(prompt)
            
            self.current_kit = kit
            if kit: self.log(f"🧩 Kit Confirmed: {kit['name']}")
            
            # C. 実装フェーズ (Phase Execution)
            for step in plan:
                self.log(f"🏗️ Phase {step['phase']}: {step['description']}")
                self._execute_phase(step, prompt, kit)
                self.ws.commit(f"Phase {step['phase']} Done")

            # D. 検証フェーズ (Runtime Check)
            self._runtime_check(kit)

            # E. 監査フェーズ (QA)
            self._final_audit()

            # F. 保存
            self.recorder.save_success(prompt, kit['name'] if kit else None, self.ws.project_files)
            
            return {
                "success": True, 
                "files": self.ws.project_files, 
                "logs": self.logs,
                "kit_used": kit['name'] if kit else None
            }

        except Exception as e:
            self.log(f"💥 Fatal Error: {e}")
            return {"success": False, "error": str(e), "logs": self.logs}
        finally:
            self.cleanup()

    # --- Sub Routines (ロジックを分離) ---

    def _execute_phase(self, phase, original_prompt, kit):
        """コード生成と静的ヒーリング（1回）"""
        target_files = phase.get('files', [])
        if not target_files: return
        
        # 構造解析
        struct_map = self.structure.analyze_project(self.ws.project_files)
        
        for target_file in target_files: # ★ここが実行のトリガーになる
            self.log(f"📝 Coding: {target_file}")
            
            # 1. 生成 (Generation)
            kit_rules = ""
            if kit: kit_rules += f"\nKit Rules: {kit.get('name')}"

            sys_prompt = f"""
            Role: Expert Developer. Task: Write code for '{target_file}'.
            Map:\n{struct_map}
            {kit_rules}
            Important: Implement FULL code. Output ONLY the code.
            """
            
            # ★ 修正済み: LLMから raw_response を取得
            raw_response = self.architect.client.generate(f"Goal: {original_prompt}\nFile: {target_file}", sys_prompt)
            
            # 2. 保存 (Save)
            # raw_response を parse_and_save_files に渡す
            new_files = self.ws.parse_and_save_files(raw_response, default_filename=target_file)
            
            # 3. 静的修復 (Static Heal) - 1回勝負
            for fname in new_files.keys():
                self._static_heal(fname, kit)

    def _static_heal(self, filename, kit):
        """静的エラー修復の1回勝負ロジック"""
        # config.MAX_RETRIES (1回) だけ回る
        for _ in range(config.MAX_RETRIES):
            # ワークスペースから最新のファイル内容を取得
            content = self.ws.project_files.get(filename, "")
            
            res = self.verifier.verify(content, filename, self.ws.project_files)
            if res['valid']: break
            
            self.log(f"🩹 Static Healing {filename}: {res['errors'][0][:50]}...")
            
            success, fixed, strategy = self.healer.heal(filename, content, res['errors'], self.ws.project_files, kit)
            
            if success and strategy not in ["Loop_Ignored", "Skipped"]:
                self.ws.save_file(filename, fixed)
            else:
                self.log(f"⚠️ Static fix skipped for {filename} ({strategy}). Proceeding.")
                break 

    def _runtime_check(self, kit):
        """ランタイムチェックとヒーリング（1回勝負）"""
        entry = next((f for f in ["app.py", "main.py"] if f in self.ws.project_files), None)
        if not entry: return

        self.log(f"🧪 Runtime Test: {entry}")
        self.runtime.install_requirements()
        
        # 1回勝負
        for _ in range(config.MAX_RETRIES):
            ok, log = self.runtime.test_run(entry)
            if ok: 
                self.log("✅ Runtime OK")
                return
            
            # 依存関係エラーなら即インストールしてリトライ
            if "ModuleNotFoundError" in log:
                missing = self._extract_module(log)
                if missing:
                    self.log(f"📦 Installing missing: {missing}")
                    self.ws.add_to_requirements(missing)
                    self.runtime.install_requirements()
                    continue

            self.log(f"💥 Runtime Error: {log[:100]}...")
            
            # ヒーリング (1回勝負)
            content = self.ws.project_files[entry]
            _, fixed, strat = self.healer.heal(entry, content, [log], self.ws.project_files, kit)
            
            if strat not in ["Loop_Ignored", "Skipped"]:
                self.ws.save_file(entry, fixed)
                self.ws.commit(f"Runtime Fix {entry}")
            else:
                self.log("⚠️ Runtime fix skipped.")
                break

    def _final_audit(self):
        """最終 QA 監査（1回）"""
        self.log("🕵️ Final QA Audit")
        res = self.qa.audit_and_fix(self.ws.project_files)
        
        if res:
            # LLMの出力からファイルをパースして保存
            self.ws.parse_and_save_files(res)
            self.ws.commit("QA Fix")
            self.log("✨ QA Fixed files")

    def _extract_module(self, log):
        import re
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", log)
        return m.group(1).split('.')[0] if m else None

    # --- Special Modes ---
    def _mode_kit_gen(self, prompt):
        yaml = self.kit_gen.generate_kit(prompt)
        name = self.kit_mgr.save_new_kit(yaml)
        return {"success": True, "logs": self.logs + [f"Kit {name} created."]}

    def _mode_research(self, prompt):
        rep = self.search.research(prompt)
        self.ws.save_file("research_report.md", rep)
        return {"success": True, "logs": self.logs + ["Research done."], "files": self.ws.project_files}


# --- Entry Point ---

def run_agent_task(prompt):
    """外部APIから呼び出されるエージェントのメイン実行関数"""
    orchestrator = Orchestrator()
    try: 
        return orchestrator.run(prompt)
    except Exception as e: 
        logger.error(f"Err: {e}")
        return {"success": False, "error": str(e), "logs": orchestrator.logs}
    finally: 
        orchestrator.cleanup()

def get_realtime_data(start=0):
    # 実行環境外ではログ取得は機能しないため、ダミーを返す
    return {"new_logs": [], "stats": {}}


## File: `requirements.txt`

txt
fastapi
uvicorn
pydantic
python-dotenv
google-generativeai
pyyaml
autopep8
beautifulsoup4
requests
ddgs
pandas
plotly
streamlit


## File: `server.py`

py
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


## File: `evo_output\ai_profiler_service.py`

py
import os
import streamlit as st
import google.generativeai as genai

class AIProfilerService:
    """
    A service class to interact with the Google Gemini AI model for
    generating analysis of development styles based on repository data.
    """
    def __init__(self, user_key=None):
        """
        Initializes the Gemini API configuration and model.
        Strictly requires a user-provided key. Does NOT fallback to environment variables
        to prevent accidental usage of the developer's quota.
        """
        # Strictly use the key provided by the user via the UI
        api_key = user_key

        # Check if the API key is provided
        if not api_key:
            st.error("Gemini API Key is missing. Please enter it in the sidebar settings.")
            st.stop()  # Halt the Streamlit application if the key is missing
        
        # Configure the generative AI library with the retrieved API key
        genai.configure(api_key=api_key)
        
        # Initialize the Gemini model.
        # 'gemini-1.5-flash' is chosen for its efficiency, speed, and capability
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')

    # @st.cache_data(ttl=3600, show_spinner="Getting AI analysis from Gemini...")
    def get_ai_analysis(self, prompt_text: str) -> str:
        """
        Calls the configured Google Gemini model to generate a comprehensive analysis
        of the development style based on the provided detailed prompt text.
        """
        try:
            # Generate content using the initialized Gemini model
            response = self.model.generate_content(prompt_text)
            
            # Validate the response from Gemini to ensure it contains content
            if not response.candidates:
                return "Gemini returned no valid candidates for analysis. " \
                       "This might indicate an issue with the prompt or the model's response."
            
            # Extract and return the generated text from the first candidate
            return response.text
        except Exception as e:
            st.error(f"An error occurred while calling the Gemini API: {e}")
            return "Failed to retrieve AI analysis due to an internal error or API issue. " \
                   "Please verify your API key, network connection, and try again."


## File: `evo_output\app.py`

py
import streamlit as st
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import re
from collections import defaultdict
import shutil
import tempfile
import subprocess

# Import services
from git_miner_service import mine_git_repository
from data_analyzer_service import DataAnalyzerService
from ai_profiler_service import AIProfilerService

# For tenacity (retry logic)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Plotting Functions ---

def plot_monthly_commits(monthly_commit_data):
    """Plots the number of commits per month."""
    if monthly_commit_data.empty:
        st.write("No commit data available to plot monthly commits.")
        return None

    fig = px.bar(
        monthly_commit_data,
        x='Month',
        y='Commits',
        title='Monthly Commit Activity',
        labels={'Commits': 'Number of Commits', 'Month': 'Month'},
        hover_data={'Month': '|%Y-%m'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_xaxes(tickformat='%Y-%m')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_activity_heatmap(activity_heatmap_data):
    """Plots a heatmap of commit activity by hour and weekday."""
    if activity_heatmap_data.empty:
        st.write("No activity heatmap data available.")
        return None

    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    all_hours = range(24)
    df_full = pd.DataFrame([(d, h) for d in weekdays for h in all_hours], columns=['Day of Week', 'Hour of Day'])
    activity_heatmap_data_merged = pd.merge(df_full, activity_heatmap_data, on=['Day of Week', 'Hour of Day'], how='left').fillna(0)
    activity_heatmap_data_merged['Commits'] = activity_heatmap_data_merged['Commits'].astype(int)

    activity_heatmap_data_merged['Day of Week'] = pd.Categorical(
        activity_heatmap_data_merged['Day of Week'],
        categories=weekdays,
        ordered=True
    )
    activity_heatmap_data_merged = activity_heatmap_data_merged.sort_values(['Day of Week', 'Hour of Day'])

    fig = px.density_heatmap(
        activity_heatmap_data_merged,
        x='Hour of Day',
        y='Day of Week',
        z='Commits',
        title='Commit Activity Heatmap (Hour of Day vs. Day of Week)',
        labels={'Hour of Day': 'Hour of Day', 'Day of Week': 'Day of Week', 'Commits': 'Number of Commits'},
        color_continuous_scale="Viridis",
        category_orders={"Day of Week": weekdays}
    )
    fig.update_xaxes(side="top", tickvals=list(range(24)))
    fig.update_layout(yaxis_autorange="reversed")
    return fig

def plot_file_extension_changes(file_extension_data):
    """Plots the distribution of file extension changes."""
    if file_extension_data.empty:
        st.write("No file extension data available.")
        return None

    fig = px.bar(
        file_extension_data.head(10),
        x='Extension',
        y='Changes',
        title='File Extension Total Changes (Top 10)',
        labels={'Extension': 'File Extension', 'Changes': 'Total Lines Changed'},
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_file_churn_ranking(file_churn_data):
    """Plots a bar chart of the top N files by churn."""
    if file_churn_data.empty:
        st.write("No file churn data available.")
        return None

    max_files = min(20, len(file_churn_data))
    if max_files == 0:
        st.write("No file churn data available to display.")
        return None
    
    top_n = st.slider(
        "Number of files to show in Churn Ranking:",
        min_value=5,
        max_value=max_files,
        value=min(10, max_files)
    )
    display_data = file_churn_data.head(top_n)

    fig = px.bar(
        display_data,
        x='churn_count',
        y='file_path',
        orientation='h',
        title=f'Top {top_n} Files by Churn (Most Frequent Changes)',
        labels={'churn_count': 'Number of Commits Affecting File', 'file_path': 'File Path'},
        color_discrete_sequence=px.colors.qualitative.Vivid,
        height=min(600, 50 * top_n + 150)
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

# --- Analysis Orchestration Function ---

# @st.cache_data(show_spinner=False)
def run_analysis(repo_url, repo_path, progress_callback, api_key):
    """
    Orchestrates the git mining, data analysis, and AI profiling.
    Uses a progress callback for Streamlit.
    """
    # Pass the user's API key to the service
    ai_profiler = AIProfilerService(user_key=api_key)
    
    results = {}
    temp_repo_dir = None

    try:
        current_repo_path = repo_path
        
        if repo_url and not repo_path:
            progress_callback(5, f"Cloning repository from {repo_url}...")
            logger.info(f"Cloning GitHub repository: {repo_url}")
            temp_repo_dir = tempfile.mkdtemp()
            
            try:
                subprocess.run(['git', 'clone', '--depth', '100', repo_url, temp_repo_dir], check=True)
                current_repo_path = temp_repo_dir
                logger.info(f"Repository cloned to temporary directory: {current_repo_path}")
            except subprocess.CalledProcessError as e:
                st.error(f"Failed to clone repository from URL: {repo_url}. Please ensure the URL is correct and the repository is public. Error: {e}")
                logger.error(f"Failed to clone repository: {e}", exc_info=True)
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred during cloning: {e}")
                logger.error(f"Cloning failed: {e}", exc_info=True)
                return None
        
        if not current_repo_path:
            st.error("No valid repository path determined for analysis.")
            return None

        # Step 1: Mine Git Repository
        progress_callback(10, "Mining Git Repository...")
        logger.info(f"Mining repository: Path={current_repo_path}")
        unique_commits_df, file_modifications_df = mine_git_repository(
            repo_path=current_repo_path,
            progress_callback=lambda current, total, msg: progress_callback(10 + int(current/total*20), msg)
        )
        
        if unique_commits_df.empty:
            st.error("No commit data found for the provided repository.")
            return None

        analyzer = DataAnalyzerService(unique_commits_df, file_modifications_df)

        # Step 2-6: Prepare Data
        progress_callback(35, "Analyzing Monthly Commits...")
        monthly_commit_data = analyzer.prepare_monthly_commit_data()
        results['monthly_commit_data'] = monthly_commit_data

        progress_callback(45, "Analyzing Commit Activity Heatmap...")
        activity_heatmap_data = analyzer.prepare_activity_heatmap_data()
        results['activity_heatmap_data'] = activity_heatmap_data

        progress_callback(60, "Analyzing File Extension Changes...")
        file_extension_data = analyzer.prepare_file_extension_data()
        results['file_extension_data'] = file_extension_data

        progress_callback(75, "Analyzing File Churn Ranking...")
        file_churn_data = analyzer.prepare_file_churn_ranking_data()
        results['file_churn_data'] = file_churn_data

        progress_callback(85, "Generating Analysis Summary for AI...")
        analysis_summary_dict = analyzer.generate_analysis_summary()
        analysis_summary = analysis_summary_dict['summary_text']
        results['analysis_summary'] = analysis_summary
        
        # Step 7: Get AI Analysis with Retry Logic
        progress_callback(90, "Calling AI for deep analysis (this may take a minute)...")
        logger.info("Calling AI profiler service...")

        # Update prompt for better actionable advice (English Version)
        enhanced_prompt = f"""
        Based on the following Git repository analysis summary, please provide a "Development Style Diagnosis" and "Concrete Advice".
        
        Analysis Summary:
        {analysis_summary}

        Please output in the following format (English):
        
        ### Development Style Diagnosis: 【(Catchy Title)】
        (Description of the style based on data)

        ### Practical Advice for Improvement
        1. **[Point 1]**: [Actionable advice]
        2. **[Point 2]**: [Actionable advice]
        3. **[Point 3]**: [Actionable advice]
        """

        ai_analysis_with_retry = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=True,
            before_sleep=lambda retry_state: logger.warning(
                f"Retrying AI analysis (attempt {retry_state.attempt_number}/{retry_state.max_attempts_reached + 1})..."
            )
        )(ai_profiler.get_ai_analysis)

        try:
            ai_summary_results = ai_analysis_with_retry(enhanced_prompt)
            results['ai_summary_results'] = ai_summary_results
        except Exception as e:
            logger.error(f"Failed to get AI analysis after multiple retries: {e}", exc_info=True)
            st.error(f"Failed to get AI analysis. Please check your API key. Error: {e}")
            results['ai_summary_results'] = "AI analysis failed."

        progress_callback(100, "Analysis complete!")
        return results

    except Exception as e:
        logger.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
        st.error(f"An unexpected error occurred during analysis: {e}")
        return None
    finally:
        if temp_repo_dir and os.path.exists(temp_repo_dir):
            try:
                shutil.rmtree(temp_repo_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary cloned repository at: {temp_repo_dir}")
            except Exception as e:
                logger.error(f"Error cleaning up temporary repository {temp_repo_dir}: {e}")

# --- Main Streamlit Application ---

def main():
    st.set_page_config(layout="wide", page_title="Git Repository AI Profiler")

    st.title("🤖 Git Repository AI Profiler")

    with st.sidebar:
        st.header("Settings")
        
        # --- API Key Input ---
        user_api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/app/apikey")
        
        st.markdown("---")
        
        st.header("Repository Input")
        repo_option = st.radio(
            "Select repository source:",
            ("Enter GitHub URL", "Enter Local Path")
        )

        repo_path_for_analysis = None
        repo_url_for_analysis = None

        if repo_option == "Enter Local Path":
            local_path_input = st.text_input("Enter path to local Git repository", value=os.getcwd())
            if os.path.isdir(local_path_input) and os.path.exists(os.path.join(local_path_input, '.git')):
                repo_path_for_analysis = local_path_input
                st.success("Valid local Git repository found.")
            elif local_path_input:
                st.warning("Invalid Git repository path.")

        elif repo_option == "Enter GitHub URL":
            github_url = st.text_input("Enter GitHub repository URL", value="https://github.com/streamlit/streamlit")
            if github_url:
                repo_url_for_analysis = github_url

        st.subheader("Analysis Controls")
        
        # Check if API Key is present
        # Strictly require user input, ignoring environment variables to prevent usage of the developer's key
        has_api_key = bool(user_api_key)
        
        if st.button("Analyze Repository", type="primary", disabled=not (bool(repo_path_for_analysis) or bool(repo_url_for_analysis))):
            if not has_api_key:
                st.error("🔒 Please enter YOUR Gemini API Key in the sidebar. This app requires your own key to function.")
            else:
                st.session_state['run_analysis'] = True
                st.session_state['repo_path_param'] = repo_path_for_analysis
                st.session_state['repo_url_param'] = repo_url_for_analysis
                st.session_state['user_api_key'] = user_api_key # Store for this session
                st.session_state['analysis_results'] = None
        
        if st.button("Clear Cache & Reset"):
            st.cache_data.clear()
            st.session_state.clear()
            st.rerun()

    # --- Main Content Area ---
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
        # Double check API key before running
        current_api_key = st.session_state.get('user_api_key')
        
        display_repo_info = st.session_state.get('repo_url_param') or st.session_state.get('repo_path_param')
        st.info(f"Starting analysis for repository: {display_repo_info}")
        
        progress_text_placeholder = st.empty()
        progress_bar_placeholder = st.progress(0)

        def update_progress_ui(percent_complete, message):
            progress_bar_placeholder.progress(int(percent_complete) / 100)
            progress_text_placeholder.text(f"Progress: {message} ({int(percent_complete)}%)")

        results = run_analysis(
            repo_url=st.session_state.get('repo_url_param'),
            repo_path=st.session_state.get('repo_path_param'),
            progress_callback=update_progress_ui,
            api_key=current_api_key # Pass the key
        )
        st.session_state['analysis_results'] = results
        st.session_state['run_analysis'] = False
        st.rerun()

    if 'analysis_results' in st.session_state and st.session_state['analysis_results'] is not None:
        results = st.session_state['analysis_results']

        st.success("Analysis Complete!")

        st.subheader("AI-Powered Repository Insights")
        if 'ai_summary_results' in results and results['ai_summary_results'] and results['ai_summary_results'] != "AI analysis failed.":
            st.markdown(results['ai_summary_results'])
        else:
            st.warning("AI analysis results are not available.")

        st.subheader("Detailed Repository Metrics")
        tab_titles = ["Monthly Commits", "Activity Heatmap", "File Extension Changes", "File Churn Ranking"]
        tabs = st.tabs(tab_titles)

        with tabs[0]:
            if 'monthly_commit_data' in results:
                st.plotly_chart(plot_monthly_commits(results['monthly_commit_data']), use_container_width=True)
        with tabs[1]:
            if 'activity_heatmap_data' in results:
                st.plotly_chart(plot_activity_heatmap(results['activity_heatmap_data']), use_container_width=True)
        with tabs[2]:
            if 'file_extension_data' in results:
                st.plotly_chart(plot_file_extension_changes(results['file_extension_data']), use_container_width=True)
        with tabs[3]:
            if 'file_churn_data' in results:
                st.plotly_chart(plot_file_churn_ranking(results['file_churn_data']), use_container_width=True)

    else:
        st.info("👈 Enter your Gemini API Key and Repository details in the sidebar to start.")

if __name__ == "__main__":
    main()


## File: `evo_output\data_analyzer_service.py`

py
import pandas as pd
from datetime import datetime
import os

# Helper function (private to this module)
def _get_file_extension(filepath: str) -> str:
    """Extracts the file extension from a given file path.
    Handles None/NaN paths and returns 'no_extension' for consistency.
    Also handles files like .gitignore correctly.
    """
    if pd.isna(filepath) or not isinstance(filepath, str):
        return 'no_extension'
    # Use os.path.splitext, then ensure it's lower case.
    # If no extension (e.g., 'file' or '.gitignore'), ext will be empty or '.gitignore' itself.
    base, ext = os.path.splitext(filepath)
    if not ext and base and base.startswith('.'): # Handles files like '.gitignore' or '.env'
        return base.lower()
    return ext.lower() if ext else 'no_extension'

class DataAnalyzerService:
    """
    Service class responsible for processing raw Git commit data
    into aggregated and summarized formats suitable for visualization
    and AI analysis.
    """

    def __init__(self, unique_commits_df: pd.DataFrame, file_modifications_df: pd.DataFrame):
        """
        Initializes the DataAnalyzerService with the raw commit and file modification data.

        Args:
            unique_commits_df: DataFrame with unique commit-level data from git_miner_service.
            file_modifications_df: DataFrame with file-level modification data from git_miner_service.
        """
        # Ensure unique_commits_df is not empty before processing
        if unique_commits_df.empty:
            self.unique_commits_df = pd.DataFrame(columns=[
                'hash', 'author_date', 'author_name', 'insertions', 'deletions',
                'lines_added_commit', 'lines_deleted_commit' # Ensure these columns are present for summary
            ])
        else:
            self.unique_commits_df = unique_commits_df.copy()
            # Ensure author_date is datetime. Convert timezone-aware to timezone-naive for consistent calculations.
            # This makes dt.weekday, dt.hour, to_period('M') behave predictably without timezone complications.
            self.unique_commits_df['author_date'] = pd.to_datetime(self.unique_commits_df['author_date']).dt.tz_localize(None)

        # Ensure file_modifications_df is not empty before processing
        if file_modifications_df.empty:
            self.file_modifications_df = pd.DataFrame(columns=[
                'commit_hash', 'change_type', 'file_path', 'lines_added', 'lines_deleted', 'extension'
            ])
        else:
            self.file_modifications_df = file_modifications_df.copy()
            # Apply robust file extension extraction using the internal helper for consistency
            self.file_modifications_df['extension'] = self.file_modifications_df['file_path'].apply(_get_file_extension)


    def prepare_monthly_commit_data(self) -> pd.DataFrame:
        """
        Prepares data for monthly commit count trend visualization, counting unique commits.

        Returns:
            DataFrame with 'Month' (datetime) and 'Commits' (count).
        """
        if self.unique_commits_df.empty:
            return pd.DataFrame(columns=['Month', 'Commits'])
        
        # Group by year and month from unique commits and count
        monthly_commits = self.unique_commits_df.groupby(
            self.unique_commits_df['author_date'].dt.to_period('M')
        ).size().reset_index(name='Commits')
        
        # Convert Period to datetime for easier plotting with Plotly
        monthly_commits['Month'] = monthly_commits['author_date'].dt.to_timestamp()
        
        return monthly_commits[['Month', 'Commits']].sort_values('Month')

    def prepare_activity_heatmap_data(self) -> pd.DataFrame:
        """
        Prepares data for activity heatmap (Day of Week vs. Hour of Day), counting unique commits.

        Returns:
            DataFrame with 'Hour of Day', 'Day of Week', 'Commits'. This is in 'long' format.
        """
        weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        if self.unique_commits_df.empty:
            # Return an empty heatmap structure consistent with plot_activity_heatmap's expectation (long format)
            return pd.DataFrame(columns=['Hour of Day', 'Day of Week', 'Commits'])

        # Extract weekday (0=Monday, 6=Sunday) and hour from unique commits
        self.unique_commits_df['weekday'] = self.unique_commits_df['author_date'].dt.weekday
        self.unique_commits_df['hour'] = self.unique_commits_df['author_date'].dt.hour

        # Group by weekday and hour, then count unique commits
        activity_counts = self.unique_commits_df.groupby(['weekday', 'hour']).size().reset_index(name='Commits')

        # Create a full grid to ensure all hours and weekdays are represented, filling missing with 0
        all_hours = pd.RangeIndex(start=0, stop=24)
        all_weekdays_nums = pd.RangeIndex(start=0, stop=7)
        full_grid = pd.MultiIndex.from_product([all_weekdays_nums, all_hours], names=['weekday', 'hour']).to_frame(index=False)
        
        # Merge with actual activity counts
        heatmap_df = pd.merge(full_grid, activity_counts, on=['weekday', 'hour'], how='left').fillna(0)
        
        # Map weekday numbers to names and rename hour column
        heatmap_df['Day of Week'] = heatmap_df['weekday'].map(lambda x: weekday_names[x])
        heatmap_df['Hour of Day'] = heatmap_df['hour']

        return heatmap_df[['Hour of Day', 'Day of Week', 'Commits']].astype({'Commits': int})


    def prepare_file_extension_data(self) -> pd.DataFrame:
        """
        Prepares data for file extension changes visualization.
        Counts total lines changed (added + deleted) per file extension.

        Returns:
            DataFrame with 'Extension', 'Changes' (total lines changed).
        """
        if self.file_modifications_df.empty:
            return pd.DataFrame(columns=['Extension', 'Changes'])
        
        # The file_modifications_df already has 'extension', 'lines_added', 'lines_deleted' for each file modification
        # Sum file-level added and deleted lines for each modification
        # Ensure columns exist and are numeric, default to 0 if not present or non-numeric
        lines_added = pd.to_numeric(self.file_modifications_df.get('lines_added', 0), errors='coerce').fillna(0)
        lines_deleted = pd.to_numeric(self.file_modifications_df.get('lines_deleted', 0), errors='coerce').fillna(0)
        
        self.file_modifications_df['file_changes_lines'] = lines_added + lines_deleted
        
        # Group by extension and sum the changes
        extension_summary = self.file_modifications_df.groupby('extension')['file_changes_lines'].sum().reset_index(name='Changes')

        # Rename columns for clarity in plots
        extension_summary.rename(columns={'extension': 'Extension'}, inplace=True)
        
        # Sort by changes
        extension_summary = extension_summary.sort_values(by='Changes', ascending=False)

        return extension_summary

    def prepare_file_churn_ranking_data(self) -> pd.DataFrame:
        """
        Calculates file churn, ranking files by the number of unique commits they appear in.

        Returns:
            DataFrame with 'file_path' and 'churn_count'.
        """
        if self.file_modifications_df.empty:
            return pd.DataFrame(columns=['file_path', 'churn_count'])

        # Group by file_path and count unique commit_hashes for each file
        file_churn = self.file_modifications_df.groupby('file_path')['commit_hash'].nunique().reset_index(name='churn_count')
        
        # Sort by churn count in descending order
        file_churn = file_churn.sort_values(by='churn_count', ascending=False)

        return file_churn


    def generate_analysis_summary(self) -> dict:
        """
        Generates a summary dictionary of key project metrics for Gemini analysis.

        Returns:
            A dictionary containing key summary statistics and a narrative summary text.
        """
        if self.unique_commits_df.empty:
            return {
                "total_commits": 0,
                "project_duration_days": 0,
                "first_commit_date": "N/A",
                "last_commit_date": "N/A",
                "total_authors": 0,
                "top_author": "N/A",
                "top_author_commits": 0,
                "avg_commits_per_day": 0.0,
                "total_lines_added": 0,
                "total_lines_deleted": 0,
                "most_active_weekday": "N/A",
                "most_active_hour": "N/A",
                "dominant_file_extensions": [],
                "top_churned_files": [],
                "summary_text": "No commit data available for analysis. Please provide a valid Git repository URL with commits."
            }

        total_commits = len(self.unique_commits_df)
        first_commit_date = self.unique_commits_df['author_date'].min()
        last_commit_date = self.unique_commits_df['author_date'].max()
        project_duration_days = (last_commit_date - first_commit_date).days if total_commits > 1 else 0

        total_authors = self.unique_commits_df['author_name'].nunique()
        author_counts = self.unique_commits_df['author_name'].value_counts()
        top_author = author_counts.index[0] if not author_counts.empty else "N/A"
        top_author_commits = int(author_counts.iloc[0]) if not author_counts.empty else 0

        avg_commits_per_day = total_commits / (project_duration_days + 1) if project_duration_days >= 0 else 0
        avg_commits_per_day = round(avg_commits_per_day, 2)

        # Use commit-level 'lines_added_commit' and 'lines_deleted_commit' for total lines changed
        # Ensure columns exist and are numeric, default to 0 if not present or non-numeric
        total_lines_added = pd.to_numeric(self.unique_commits_df.get('lines_added_commit', 0), errors='coerce').fillna(0).sum()
        total_lines_deleted = pd.to_numeric(self.unique_commits_df.get('lines_deleted_commit', 0), errors='coerce').fillna(0).sum()


        # Get activity heatmap data to find most active weekday/hour
        heatmap_data = self.prepare_activity_heatmap_data() # Call internal method
        most_active_weekday = "N/A"
        most_active_hour = "N/A"
        if not heatmap_data.empty and heatmap_data['Commits'].sum() > 0:
            # Find the row with maximum 'Commits'
            max_activity_row = heatmap_data.loc[heatmap_data['Commits'].idxmax()]
            most_active_weekday = max_activity_row['Day of Week']
            most_active_hour = int(max_activity_row['Hour of Day'])


        # Get file extension data
        file_extension_summary = self.prepare_file_extension_data() # Call internal method
        dominant_file_extensions = file_extension_summary.head(3)['Extension'].tolist() if not file_extension_summary.empty else []

        # Get file churn data
        file_churn_summary = self.prepare_file_churn_ranking_data()
        top_churned_files = file_churn_summary.head(3)['file_path'].tolist() if not file_churn_summary.empty else []


        # Construct summary text
        summary_lines = []
        summary_lines.append(f"This repository contains {total_commits} unique commits.")
        if total_commits > 0:
            summary_lines.append(f"It spans {project_duration_days} days, from {first_commit_date.strftime('%Y-%m-%d')} to {last_commit_date.strftime('%Y-%m-%d')}.")
            summary_lines.append(f"There are {total_authors} unique authors. The most active author is '{top_author}' with {top_author_commits} commits.")
            summary_lines.append(f"On average, {avg_commits_per_day} commits are made per day.")
            summary_lines.append(f"A total of {int(total_lines_added)} lines were added and {int(total_lines_deleted)} lines were deleted across all unique commits.")
            if most_active_weekday != "N/A":
                summary_lines.append(f"The peak activity time is typically on {most_active_weekday} around {most_active_hour}:00.")
            if dominant_file_extensions:
                summary_lines.append(f"Dominant file types changed include: {', '.join(dominant_file_extensions)}.")
            else:
                summary_lines.append("No specific dominant file extensions were identified.")
            if top_churned_files:
                summary_lines.append(f"Top churned files (frequently changed) include: {', '.join(top_churned_files)}.")
            else:
                summary_lines.append("No specific churned files were identified.")
        
        summary_text = " ".join(summary_lines)

        summary_data = {
            "total_commits": total_commits,
            "project_duration_days": project_duration_days,
            "first_commit_date": str(first_commit_date.strftime('%Y-%m-%d %H:%M:%S')), # Format for consistent JSON
            "last_commit_date": str(last_commit_date.strftime('%Y-%m-%d %H:%M:%S')),   # Format for consistent JSON
            "total_authors": total_authors,
            "top_author": top_author,
            "top_author_commits": top_author_commits,
            "avg_commits_per_day": avg_commits_per_day,
            "total_lines_added": int(total_lines_added), # Ensure int for JSON serialization
            "total_lines_deleted": int(total_lines_deleted), # Ensure int for JSON serialization
            "most_active_weekday": most_active_weekday,
            "most_active_hour": most_active_hour,
            "dominant_file_extensions": dominant_file_extensions,
            "top_churned_files": top_churned_files, # Added for completeness in summary data
            "summary_text": summary_text
        }
        return summary_data


## File: `evo_output\gemini_service.py`

py
import os
import google.generativeai as genai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
import logging

# Configure logging for tenacity and Gemini service
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GeminiService:
    """
    A service class to interact with the Google Gemini API,
    incorporating retry logic for robust API calls.
    """

    def __init__(self):
        """
        Initializes the GeminiService by configuring the API key
        and loading the Gemini Pro model.
        Raises a ValueError if the GEMINI_API_KEY environment variable is not set.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set. Please set it to use Gemini.")
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        logger.info("GeminiService initialized and model 'gemini-pro' loaded.")

    @retry(
        wait=wait_random_exponential(multiplier=1, min=4, max=10), # Exponential backoff with random jitter
        stop=stop_after_attempt(3),                              # Stop after 3 attempts
        retry=retry_if_exception_type(Exception),                 # Retry on any exception
        reraise=True                                             # Re-raise the last exception if all retries fail
    )
    def generate_content_with_retry(self, prompt: str) -> str:
        """
        Sends a prompt to the Gemini API and retrieves the generated content,
        with built-in retry logic using tenacity.

        Args:
            prompt: The text prompt to send to the Gemini model.

        Returns:
            The generated text content from the Gemini model.

        Raises:
            Exception: If the Gemini API call fails after all retries,
                       or if the response content is empty/invalid.
        """
        logger.info(f"Attempting Gemini content generation (attempt {self.generate_content_with_retry.retry.statistics['attempts'] + 1} of 3). Prompt snippet: {prompt[:100]}...")
        try:
            response = self.model.generate_content(prompt)

            # Check if the response contains valid parts and text
            if not response.parts or not response.text:
                logger.warning(f"Gemini API returned an empty or invalid response. Response: {response}")
                # Raise an error to trigger a retry if applicable
                raise ValueError("Gemini API returned no content or invalid response.")

            logger.info("Gemini content generation successful.")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed with error: {e}")
            # The 'reraise=True' in @retry decorator will handle re-raising after all attempts.
            # We just need to ensure an exception is raised here to signal failure to tenacity.
            raise


## File: `evo_output\git_miner_service.py`

py
import os
from pydriller import Repository, Git
import pandas as pd
from datetime import datetime

def mine_git_repository(repo_path: str, progress_callback=None) -> (pd.DataFrame, pd.DataFrame):
    """
    Mines a Git repository to extract commit and file change data.

    This function iterates through all commits in a specified Git repository,
    collecting detailed information about each commit and the files modified
    within them. It can optionally report progress via a callback function,
    which is useful for UI updates (e.g., Streamlit progress bars).

    Args:
        repo_path (str): The absolute or relative path to the Git repository.
                         This path should point to the root directory of the repository.
        progress_callback (callable, optional): A function to call with progress updates.
                                                If provided, it should accept three arguments:
                                                (current_step: int, total_steps: int, message: str).
                                                Defaults to None, meaning no progress reporting.

    Returns:
        tuple: A tuple containing two pandas DataFrames:
               - commits_df (pd.DataFrame): DataFrame of commit-level data, with columns
                                            like 'hash', 'author_name', 'author_date', 'message',
                                            'lines_added_commit', 'lines_deleted_commit',
                                            'files_changed_commit'.
               - file_changes_df (pd.DataFrame): DataFrame of file-level change data, with columns
                                                 like 'commit_hash', 'change_type', 'old_path',
                                                 'new_path', 'file_path', 'lines_added',
                                                 'lines_deleted', 'nloc', 'complexity'.

    Raises:
        FileNotFoundError: If the specified repository path does not exist.
        ValueError: If the specified path is not a valid Git repository.
    """
    if not os.path.exists(repo_path):
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not os.path.isdir(os.path.join(repo_path, '.git')):
        raise ValueError(f"'{repo_path}' is not a valid Git repository.")

    commits_data = []
    file_changes_data = []

    # Attempt to get the total number of commits for an accurate progress bar
    try:
        git_helper = Git(repo_path)
        total_commits = git_helper.total_commits()
    except Exception as e:
        # Fallback if `Git().total_commits()` fails (e.g., repository corrupted, no commits)
        print(f"Warning: Could not get total commit count for repository {repo_path}: {e}")
        total_commits = 0  # Indicate unknown total

    current_commit_count = 0

    # Initialize Repository miner
    repo_miner = Repository(repo_path)

    for commit in repo_miner.traverse_commits():
        current_commit_count += 1

        # Report progress if a callback is provided
        if progress_callback:
            # If total_commits is unknown (0), we estimate total to prevent ZeroDivisionError
            # and allow the progress bar to show relative movement.
            display_total = total_commits if total_commits > 0 else current_commit_count + 1
            progress_callback(
                current_commit_count,
                display_total,
                f"Mining commit {commit.hash[:7]} by {commit.author.name}"
            )

        # Collect commit-level data
        commits_data.append({
            'hash': commit.hash,
            'author_name': commit.author.name,
            'author_email': commit.author.email,
            'author_date': commit.author_date,
            'committer_name': commit.committer.name,
            'committer_email': commit.committer.email,
            'committer_date': commit.committer_date,
            'message': commit.msg,
            'lines_added_commit': commit.insertions,
            'lines_deleted_commit': commit.deletions,
            'files_changed_commit': len(commit.modified_files)
        })

        # Collect file-level change data for each modification in the commit
        for mod in commit.modified_files:
            file_changes_data.append({
                'commit_hash': commit.hash,
                'change_type': mod.change_type.name,  # e.g., ADD, DELETE, MODIFY, RENAME, COPY
                'old_path': mod.old_path,
                'new_path': mod.new_path,
                # 'file_path' represents the path of the file *after* the change.
                # For deleted files, new_path is None, so old_path is used.
                'file_path': mod.new_path if mod.new_path else mod.old_path,
                'lines_added': mod.added_lines,
                'lines_deleted': mod.deleted_lines,
                # Pydriller can return None for nloc/complexity, default to 0 for int compatibility
                'nloc': mod.nloc if mod.nloc is not None else 0,
                'complexity': mod.complexity if mod.complexity is not None else 0
            })

    # Convert collected data into pandas DataFrames
    commits_df = pd.DataFrame(commits_data)
    file_changes_df = pd.DataFrame(file_changes_data)

    # Post-processing for commits_df
    if not commits_df.empty:
        # Ensure author_date and committer_date are timezone-aware datetime objects, converted to UTC
        commits_df['author_date'] = pd.to_datetime(commits_df['author_date'], utc=True)
        commits_df['committer_date'] = pd.to_datetime(commits_df['committer_date'], utc=True)
    else:
        # Define empty DataFrame with correct dtypes if no commits were found
        commits_df = pd.DataFrame(columns=[
            'hash', 'author_name', 'author_email', 'author_date',
            'committer_name', 'committer_email', 'committer_date', 'message',
            'lines_added_commit', 'lines_deleted_commit', 'files_changed_commit'
        ]).astype({
            'hash': str, 'author_name': str, 'author_email': str,
            'author_date': 'datetime64[ns, UTC]', 'committer_name': str,
            'committer_email': str, 'committer_date': 'datetime64[ns, UTC]',
            'message': str, 'lines_added_commit': int,
            'lines_deleted_commit': int, 'files_changed_commit': int
        })

    # Post-processing for file_changes_df
    if not file_changes_df.empty:
        # Fill potential None values in path columns with empty strings for consistency
        file_changes_df['old_path'] = file_changes_df['old_path'].fillna('').astype(str)
        file_changes_df['new_path'] = file_changes_df['new_path'].fillna('').astype(str)
        file_changes_df['file_path'] = file_changes_df['file_path'].fillna('').astype(str)
    else:
        # Define empty DataFrame with correct dtypes if no file changes were found
        file_changes_df = pd.DataFrame(columns=[
            'commit_hash', 'change_type', 'old_path', 'new_path', 'file_path',
            'lines_added', 'lines_deleted', 'nloc', 'complexity'
        ]).astype({
            'commit_hash': str, 'change_type': str, 'old_path': str,
            'new_path': str, 'file_path': str, 'lines_added': int,
            'lines_deleted': int, 'nloc': int, 'complexity': int
        })

    return commits_df, file_changes_df


## File: `evo_output\requirements.txt`

txt
streamlit
PyDriller
plotly
pandas
google-generativeai
tenacity


## File: `evo_output\新規 テキスト ドキュメント.txt`

txt
# 🤖 Git Repository AI Profiler

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by Gemini](https://img.shields.io/badge/AI-Gemini%20Flash-8E75B2)](https://deepmind.google/technologies/gemini/)

> **"Not just a log analyzer. It's your AI Career Coach."**
>
> Reveal your coding style, detect burnout risks, and get actionable advice from an AI CTO based on your Git history.

![Demo App Screenshot](https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/_static/logo.png)
*(Replace this link with your actual screenshot or GIF)*

## 🧐 What is this?

**Git Repository AI Profiler** is a Streamlit application that mines your Git repository meta-data and uses **Google Gemini 2.5 Flash** to profile your development habits.

Instead of boring statistics, it gives you a **"Developer Persona"** (e.g., *"The Midnight Sprinter"*, *"The Weekend Warrior"*) and provides **concrete, sometimes harsh, advice** to improve your code quality and work-life balance.

## ✨ Key Features

* **📊 Interactive Visualizations**:
    * **Monthly Commits**: Track your productivity trends.
    * **Activity Heatmap**: Visualize your peak coding hours (Day vs Hour).
    * **Churn Ranking**: Identify "High-Risk Files" that are modified too frequently.
* **🧠 AI-Powered Profiling**:
    * Generates a unique **"Dev Persona"** based on your commit patterns.
    * Provides **CTO-level advice** on refactoring, burnout prevention, and architectural improvements.
* **⚡ High Performance**:
    * Real-time progress tracking for long mining tasks.
    * Robust API handling with automatic retries.
* **📱 PWA Ready**:
    * Installable on mobile devices as a Progressive Web App.

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR_USERNAME/git-repo-ai-profiler.git](https://github.com/YOUR_USERNAME/git-repo-ai-profiler.git)
cd git-repo-ai-profiler
2. Set up the environment
It is recommended to use a virtual environment.

Bash

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate
3. Install dependencies
Bash

pip install -r requirements.txt
4. Set your API Key
You need a Google Gemini API Key. Get it from Google AI Studio.

Windows (Command Prompt):

DOS

set GOOGLE_API_KEY=your_api_key_here
Mac/Linux:

Bash

export GOOGLE_API_KEY="your_api_key_here"
5. Run the App!
Bash

streamlit run app.py
🛠️ Tech Stack
Frontend: Streamlit

Data Processing: Pandas, PyDriller

Visualization: Plotly

AI Engine: Google Gemini API (via google-generativeai)

Resilience: Tenacity

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the project

Create your Feature Branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some AmazingFeature')

Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

📄 License
Distributed under the MIT License. See LICENSE for more information.

Created with ❤️ and AI by [Emma Saka]


## File: `kits\Video Content Repurposer Engine.yaml`

yaml
id: "video_content_repurposer"
name: "Video to Blog Engine"
description: "YouTube動画から音声を抽出し、Gemini 1.5 Flashで文字起こしを行い、ブログ記事とSNS投稿文を自動生成するStreamlitアプリ。"
version: "1.0.0"

triggers:
  keywords: ["YouTube", "文字起こし", "要約", "ブログ自動生成", "Streamlit"]
  sample_prompts:
    - "YouTube動画をブログ記事にするツールを作って"
    - "動画のURLから要約とツイートを作るアプリ"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "yt-dlp (Video/Audio Downloader)"
    - "google-generativeai (Gemini 2.5 Flash for Transcription & Writing)"
    - "Streamlit (Web UI)"
    - "pydub (Audio Processing)"

  core_components:
    - "Media Service (Download & Convert)"
    - "AI Service (Transcribe & Generate)"
    - "UI Controller (Streamlit View)"
    - "Main Entry Point"

  expected_file_structure:
    - "app.py"             # UIのエントリーポイント
    - "media_service.py"   # ダウンロード・音声変換ロジック
    - "ai_service.py"      # Gemini APIとの通信ロジック
    - "utils.py"           # 共通ユーティリティ
    - "requirements.txt"
    - "temp/"              # 一時保存フォルダ

resources:
  domain_knowledge: |
    【開発の鉄則（絶対厳守）】
    1. **メインファイルの役割制限 (Strict Rule):**
       - `app.py` には **UIの表示コード（ボタン、テキストボックスなど）以外を書いてはならない**。
       - 具体的な処理（ダウンロード、API呼び出し）は、必ず `media_service.py` や `ai_service.py` をインポートして呼び出すこと。
       - ロジックのベタ書きは「スパゲッティコード」とみなし禁止する。
    
    2. **大容量データの処理:**
       - 動画の音声データは大きいため、一度ローカルの `temp/` フォルダに `.mp3` として保存してからAPIにアップロードする設計にすること。
       - Gemini 2.5 Flash は音声ファイルを直接扱えるため、Whisperではなく **Gemini APIのFile API** を使用すること（処理が速く簡単）。
    
    3. **エラーハンドリング:**
       - `yt-dlp` はURLが無効だとエラーを吐くため、`try-except` で囲み、ユーザーに分かりやすいエラーメッセージをUIに表示すること。
    
    4. **コスト意識:**
       - 文字起こしや要約には、安価で長文に強い **Gemini 2.5 Flash** モデルを指定すること。


## File: `kits\autonomous_research_agent.yaml`

yaml
id: "autonomous_research_agent"
name: "Deep Research Agent (Perplexity Style)"
description: "Web検索、スクレイピング、要約を自律的に行い、出典付きの調査レポートを作成するエージェント。"
version: "1.0.0"

triggers:
  keywords: ["リサーチ", "検索", "調査", "レポート", "Perplexity"]
  sample_prompts:
    - "最新のAIトレンドについて調べてレポートにして"
    - "競合他社の動向を調査するエージェントを作って"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "duckduckgo-search (Search Tool)"
    - "trafilatura (Robust Text Extraction)"
    - "Gemini 2.5 Flash (Reading & Summarizing)"
    - "Gemini 2.5 Pro (Final Report Writing)"
    - "MdUtils (Markdown Generation)"

  core_components:
    - "Search Engine (Query Handler)"
    - "Web Scraper (Content Fetcher)"
    - "Information Synthesizer (LLM)"
    - "Report Builder"
    - "Main Controller"

  expected_file_structure:
    - "main.py"
    - "search_service.py"
    - "scrape_service.py"
    - "synthesis_service.py"
    - "config.py"
    - "requirements.txt"
    - "output_report.md" # 生成される成果物

resources:
  domain_knowledge: |
    【開発の鉄則（絶対遵守）】
    1. **メインファイルの役割放棄 (Strict Rule):**
       - `main.py` はオーケストレーター（指揮者）に徹すること。
       - 検索、スクレイピング、要約などの具体的ロジックを `main.py` に書くことは**厳禁**とする。必ず各 `_service.py` をインポートして呼び出せ。
    
    2. **スクレイピングの防御力:**
       - Webサイトはアクセス拒否（403）やタイムアウトが頻発する。
       - `scrape_service.py` 内では必ず `try-except` でエラーを握り潰し、**「1つのサイトがダメでも止まらずに次へ行く」**構造にせよ。
       - 取得したテキストは長すぎる場合があるため、先頭 10,000 文字でトリミングする処理を入れること。
    
    3. **情報の出典管理:**
       - 検索結果の URL と タイトル は最後まで保持し、最終レポートの末尾に「参考文献リスト」として記載すること。
    
    4. **段階的処理:**
       - 一気にやろうとしないこと。
       - Step 1: 検索してURLリストを得る
       - Step 2: 各URLからテキストを抜く
       - Step 3: テキストを要約する
       - Step 4: 要約を統合して執筆する
       - このフローを `main.py` で順序よく実行せよ。


## File: `kits\chrome_extension_expert.yaml`

yaml
id: "chrome_extension_expert"
name: "Chrome Extension Expert (Manifest V3)"
description: "ミスを許さない厳格なChrome拡張機能開発キット。HTML/JSの整合性を重視。"
version: "2.0.0"

triggers:
  keywords:
    - "chrome extension"
    - "クローム拡張"
    - "プラグイン"
    - "manifest v3"
  sample_prompts:
    - "Chrome拡張機能を作って"
    - "ブラウザの表示を変える拡張機能"

blueprint:
  suggested_tech_stack:
    - "HTML5 / CSS3"
    - "JavaScript (ES6+)"
    - "Manifest V3"
  
  core_components:
    - "Popup UI"
    - "Background Service Worker"
    - "Content Scripts"
    - "Message Passing"

  # ★重要: 作成順序を強制する（HTMLが先！）
  expected_file_structure:
    - "manifest.json"
    - "popup/popup.html"  # 先にUIを作る
    - "popup/popup.js"    # 次にロジック
    - "popup/popup.css"
    - "background.js"
    - "icons/icon16.png" # (ダミー)

resources:
  domain_knowledge: |
    【開発の絶対ルール (Strict Rules)】
    
    1. **順序の厳守:**
       - 必ず `popup.html` を作成してから、その後に `popup.js` を実装すること。
       - AIは「存在しないID」を捏造しがちなので、JSを書く際は直前に作成したHTMLのIDを確認すること。

    2. **IDの整合性チェック (Anti-Hallucination):**
       - `popup.js` で `document.getElementById('xyz')` を使う場合、そのID `'xyz'` が `popup.html` に実在することを100%保証すること。
       - もしHTMLにない場合は、JS側で勝手に参照せず、HTML側にIDを追加する修正案を出すこと。
    
    2-b. **事前スキャンと自動ID補正 (ID Auto-Sync):**
       - AIは `popup.js` を生成する前に必ず `popup.html` 内の全ての `id` 属性をスキャンし、参照可能なIDリストを内部に保持すること。
       - JS側で存在しないIDを参照しようとした場合、勝手にJSに書き込む前に自動で `popup.html` に該当IDを追加する修正案を生成してからJSを作成すること。
       - このルールに従うことで、HTMLとJS間の整合性を100%自動保証する。

    3. **Manifest V3の罠回避:**
       - `background` は `scripts` ではなく `"service_worker"` を使うこと。
       - `browser_action` ではなく `"action"` を使うこと。
       - `Content Security Policy (CSP)` エラーを防ぐため、インラインスクリプト（HTML内の `<script>...code...</script>`）は禁止。必ず外部ファイル (`popup.js`) に分離すること。

    4. **非同期通信の鉄則:**
       - `chrome.runtime.onMessage` リスナー内で非同期処理（`sendResponse`）を行う場合は、必ず `return true;` を記述すること。これを忘れるとメッセージチャネルが即座に閉じられる。


## File: `kits\data_analysis_dashboard.yaml`

yaml
id: "data_analysis_dashboard"
name: "Data Analysis & Visualization Dashboard"
description: "CSV/Excelデータをアップロードし、自動で集計・可視化を行う分析ダッシュボード"
version: "1.0.0"

triggers:
  keywords:
    - "データ"
    - "分析"
    - "グラフ"
    - "可視化"
    - "csv"
    - "dashboard"
    - "plot"
  sample_prompts:
    - "CSVファイルを読み込んでグラフにするアプリを作って"
    - "売上データを分析するダッシュボードが欲しい"
    - "データの傾向を可視化するツール"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+ (Flask)"
    - "Pandas (Data Processing)"
    - "Plotly / Matplotlib (Visualization)"
    - "Tailwind CSS (UI)"
    - "Chart.js / Plotly.js (Frontend Rendering)"

  core_components:
    - "File Uploader (CSV/Excel)"
    - "Data Processor (Pandas DataFrame)"
    - "Statistical Analyzer (Mean, Median, Corr)"
    - "Chart Generator API"

  expected_file_structure:
    - "app.py"
    - "analysis_logic.py"
    - "templates/index.html"
    - "requirements.txt"
    - "static/js/dashboard.js"

resources:
  domain_knowledge: |
    【データ分析アプリ開発の鉄則】
    1. **データ処理とWeb表示の分離:**
       - 重いデータ処理は `analysis_logic.py` 内で行い、`Pandas` をフル活用すること。
       - `app.py` はデータの受け渡しとAPI提供に徹すること。
    
    2. **ファイルアップロード処理:**
       - ユーザーがCSVファイルをアップロードする機能を必ず実装すること。
       - アップロードされたファイルは `pd.read_csv()` で読み込み、データフレームとして扱う。
       - セキュリティのため、ファイル保存時は `werkzeug.utils.secure_filename` を使用するか、一時ファイル (`tempfile`) で処理すること。

    3. **可視化のアプローチ:**
       - バックエンドでグラフ画像を生成してBase64で返す方法（Matplotlib）か、データをJSONで返してフロントエンドで描画する方法（Chart.js/Plotly.js）のどちらかを選択する。
       - **推奨:** インタラクティブ性を高めるため、データをJSONで返し、フロントエンド(`Chart.js` または `Plotly.js`)で描画するパターンを優先すること。

    4. **エラーハンドリング:**
       - 読み込めないフォーマットや、空のデータが送られた場合の例外処理 (`try-except`) を必ず入れること。
       - 数値データ以外の列が含まれている場合の処理（除外するか、カウントするか）を考慮すること。


## File: `kits\git_profiler.yaml`

yaml
id: "git_repo_profiler"
name: "Git Repository AI Profiler"
description: "Gitリポジトリを解析し、コミット履歴やコード変更量から開発スタイル、貢献パターン、コードの健全性を可視化＆AI診断するダッシュボード。"
version: "1.0.0"

triggers:
  keywords: ["Git", "分析", "可視化", "リポジトリ", "プロフィール", "PyDriller"]
  sample_prompts:
    - "このGitHubリポジトリの活動履歴を分析してレポートにして"
    - "開発チームのコードコミット傾向を可視化するツールを作って"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "PyDriller (Git Mining)"
    - "Pandas (Data Analysis)"
    - "Plotly (Interactive Charts)"
    - "Streamlit (Dashboard UI)"
    - "google-generativeai (Gemini 1.5 Flash)"

  core_components:
    - "Git Miner (Commit History Extractor)"
    - "Data Analyzer (Statistics & Metrics)"
    - "AI Profiler (Insights Generation)"
    - "Dashboard (Visualizations)"

  expected_file_structure:
    - "app.py"             # Streamlit Entry Point
    - "miner_service.py"   # PyDriller logic (Repo cloning & mining)
    - "analysis_service.py"# Pandas logic (Aggregations)
    - "ai_service.py"      # Gemini logic (Profile generation)
    - "charts.py"          # Plotly visualization logic
    - "requirements.txt"

resources:
  domain_knowledge: |
    【開発の鉄則】
    1. **データマイニングの分離とキャッシュ:**
       - リポジトリ解析は重いため、`miner_service.py` で `pydriller` を実行し、結果（コミット日時、著者、変更行数、ファイル名）を抽出する。
       - Streamlitの `st.cache_data` を使用して、同じリポジトリの再解析を防ぐこと。
    
    2. **「活動ヒートマップ」の実装:**
       - GitHubの草（Contributions）のようなグラフだけでなく、`Plotly` を使って「曜日 × 時間帯」のヒートマップ（Punch Card）を作成せよ。これにより「深夜稼働率」などを可視化できる。
    
    3. **AIによる「開発者性格診断」:**
       - 集計データ（例: 平均コミットメッセージ長、修正頻度、活動時間帯）をテキスト化してGeminiに渡し、「この開発チームの強みと健康状態」をユーモラスかつ鋭く分析させること。
    
    4. **安全なクローン:**
       - リポジトリは Python の `tempfile` モジュールを使って一時ディレクトリにクローンし、分析後は確実に削除（クリーンアップ）する設計にすること。
    【厳格な責務分離ルール】
    
     app.py:
       - Streamlit UIのみ（st.title, st.text_input, st.buttonなど）
       - 他のサービスをインポートして使用
       - ロジックを書かない


## File: `kits\jra_keiba.yaml`

yaml
id: "jra_racing_prediction"
name: "JRA Horse Racing AI Predictor"
description: "JRA-VANデータ構造に対応した競馬予想・分析システム"
version: "2.1.0"

triggers:
  keywords: ["競馬", "jra", "予想", "馬券", "keiba", "回収率"]
  sample_prompts:
    - "週末の重賞レースを予想するAIを作って"
    - "血統データを分析して穴馬を見つけるスクリプト"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "pandas (データ加工)"
    - "scikit-learn / LightGBM (機械学習)"
    - "BeautifulSoup (補助的なスクレイピング)"

  core_components:
    - "Data Preprocessing Pipeline (欠損値処理・カテゴリ変換)"
    - "Feature Engineering (血統・騎手・過去走破タイム)"
    - "Backtesting Engine (回収率シミュレーション)"

  expected_file_structure:
    - "main.py"
    - "data_loader.py"
    - "model.py"
    - "strategies/bloodline.py"

resources:
  domain_knowledge: |
    【重要】日本の競馬データ分析における鉄則:
    1. 「馬場状態（良・稍重・重・不良）」は最重要特徴量の一つ。必ず考慮すること。
    2. 血統データ（父・母父）はカテゴリ変数として扱うより、ターゲットエンコーディングが有効。
    3. タイム指数（Speed Index）を計算する際は、競馬場ごとの基準タイム差を補正すること。
    4. 3連単の予測はノイズが多いため、まずは「複勝圏内（3着以内）」の確率予測モデルを推奨。


## File: `kits\mahjong_pro.yaml`

yaml
id: "mahjong_browser_game"
name: "Mahjong Browser Game (Full Stack)"
description: "Pythonの判定ロジックと、JSのゲーム進行を組み合わせた、実際に遊べる一人麻雀"
version: "3.0.0"

triggers:
  keywords: ["麻雀", "mahjong", "ゲーム", "プレイ"]
  sample_prompts:
    - "ブラウザで遊べる麻雀ゲームを作って"
    - "一人麻雀アプリ"

blueprint:
  suggested_tech_stack:
    - "Python 3.10 (Flask Backend)"
    - "Vanilla JavaScript (Frontend Logic)"
    - "Tailwind CSS (UI Design)"
    - "mahjong (Library: https://pypi.org/project/mahjong/)"

  core_components:
    - "Game Loop (Init -> Draw -> Discard -> Check -> Repeat)"
    - "Shanten Calculation API"
    - "Visual Tile Rendering (CSS/Unicode)"

  expected_file_structure:
    - "app.py"
    - "templates/index.html"  # これがないと始まらない
    - "requirements.txt"

resources:
  domain_knowledge: |
    【開発の絶対ルール】
    1. **「計算機」ではなく「ゲーム」を作ること。**
       - ユーザーが手入力するのではなく、プログラムがランダムに配牌すること。
       - 「ツモボタン」と「クリックで捨て牌」の機能を実装すること。
    
    2. **ライブラリの正しい使い方 (Copy this!)**
       - `mahjong` ライブラリでシャンテン数を計算する際は、以下のコードスニペットを厳守すること。
       (古い `mahjong.hand` は存在しないため使わないこと)
       
       ```python
       from mahjong.shanten import Shanten
       from mahjong.tile import TilesConverter
       
       # 13枚または14枚の手牌リスト (例: ['1m', '2m'...]) を受け取る
       tiles_34 = TilesConverter.to_34_array(tiles_list)
       calculator = Shanten()
       result = calculator.calculate_shanten(tiles_34)
       # result: -1=Agari, 0=Tenpai, 1=1-Shanten...
       ```

    3. **フロントエンドの挙動**
       - `index.html` 内に `<script>` タグでゲームロジック（山牌管理、手牌配列）を書くこと。
       - 牌を捨てるたびに `fetch('/api/check', ...)` でバックエンドに問い合わせて、シャンテン数を表示すること。


## File: `kits\ml_model_api_wrapper.yaml`

yaml
id: "ml_model_api_wrapper"
name: "Machine Learning Model API Wrapper"
description: "既存の学習済みモデル（例: scikit-learn, PyTorch）をロードし、REST APIとして公開するAPIキット。"
version: "1.0.0"

triggers:
  keywords:
    - "機械学習"
    - "AIモデル"
    - "予測"
    - "API公開"
    - "scikit-learn"
  sample_prompts:
    - "学習済みモデルをAPIとして公開したい"
    - "ユーザーの入力から予測結果を返すWebサービスを作って"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+ (FastAPI/Flask)"
    - "Scikit-learn / TensorFlow (Model Library)"
    - "Numpy (データ処理)"
    - "FastAPI (API提供。Flaskより高速でモダンなため推奨)"

  core_components:
    - "Model Loader (pickle, joblib)"
    - "Data Validator (入力の型チェック)"
    - "Prediction Endpoint (/predict)"

  expected_file_structure:
    - "main.py"
    - "model_wrapper.py"
    - "model.pkl" # (ダミーファイルとして記載)
    - "requirements.txt"

resources:
  domain_knowledge: |
    【MLモデルAPIのベストプラクティス】
    1. **Model Loading:** モデルはメインループ外（起動時）に一度だけロードし、メモリにキャッシュすること（推論速度のため）。
    2. **Input Validation:** APIの入力データは必ずNumpy配列に変換し、形状（shape）をチェックすること。
    3. **FastAPIの使用:** 予測サービスは低遅延が求められるため、FastAPIの非同期処理を優先すること。
    4. **セキュリティ:** モデルファイル（.pklなど）は直接アップロードさせず、コードで参照させること。


## File: `kits\smart_bartender.yaml`

yaml
id: "smart_bartender_ai"
name: "Smart Bartender & Cabinet Manager"
description: "自宅にあるお酒やジュース（在庫）を登録し、それらで作れるカクテルをGeminiが提案・創作するアプリ。SQLiteで在庫管理を行う。"
version: "1.0.0"

triggers:
  keywords: ["カクテル", "バーテンダー", "お酒", "レシピ生成", "在庫管理"]
  sample_prompts:
    - "今あるお酒で作れるカクテルを教えて"
    - "ウイスキーと炭酸水を使ったアレンジレシピを考えて"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "Streamlit (Web UI)"
    - "google-generativeai (Gemini 2.5 Flash)"
    - "SQLAlchemy (Inventory DB)"
    - "Pydantic (Recipe Schema Validation)"

  core_components:
    - "Inventory Manager (CRUD for Liquors/Mixers)"
    - "Bartender Brain (LLM Recipe Generator)"
    - "UI Controller"
    - "Database Model"

  expected_file_structure:
    - "app.py"             # UIエントリーポイント
    - "inventory_service.py" # 在庫管理ロジック
    - "bartender_service.py" # Geminiとの対話ロジック
    - "models.py"          # DBとPydanticのモデル定義
    - "database.py"        # DB接続周り
    - "requirements.txt"

resources:
  domain_knowledge: |
    【開発の鉄則】
    1. **データの構造化 (Strict JSON Output):**
       - AIが生成するレシピは、必ずJSON形式で受け取り、Pydanticモデルでパースすること。
       - 必要なフィールド: `name` (名前), `ingredients` (材料リスト), `instructions` (手順), `flavor_profile` (味の特徴), `alcohol_strength` (度数目安)。
       - `bartender_service.py` 内で `response_schema` を使用してGeminiに強制すること。
    
    2. **在庫ベースの思考:**
       - プロンプトには必ず「現在の在庫リスト（Inventory）」を含め、「これに含まれる材料を優先的に使用せよ」と指示すること。
       - 足りない材料がある場合は、「あとこれがあれば作れます」という提案（Missing Ingredients）を含めるロジックにすると親切。
    
    3. **UI/UX:**
       - Streamlitの `st.data_editor` を使って、在庫の追加・削除を直感的に行えるようにすること。
       - 生成されたカクテルはカード形式で見やすく表示すること。


## File: `kits\social_bot_kit.yaml`

yaml
id: "social_bot_automation"
name: "Social Media Auto-Poster (Selenium)"
description: "Google Sheetsからデータを読み込み、Seleniumを使ってX/Threadsに自動投稿するBot"
version: "1.0.0"

triggers:
  keywords: ["自動投稿", "SNS", "bot", "selenium", "x", "threads"]
  sample_prompts:
    - "スプレッドシートの内容をXとThreadsに投稿するツールを作って"
    - "SNS自動投稿ツール"

blueprint:
  suggested_tech_stack:
    - "Python 3.10"
    - "Selenium (Browser Automation)"
    - "gspread (Google Sheets API)"
    - "schedule (Task Scheduling)"
    - "oauth2client (Auth)"

  core_components:
    - "Sheets Loader (Read time & content)"
    - "Browser Manager (Headless Chrome)"
    - "X Poster Logic (XPath selector)"
    - "Threads Poster Logic"
    - "Scheduler Loop"

  expected_file_structure:
    - "main.py"
    - "poster_logic.py"
    - "sheets_handler.py"
    - ".env"
    - "requirements.txt"

resources:
  domain_knowledge: |
    【開発の鉄則】
    1. **APIではなくSeleniumを使う:**
       - XとThreadsはAPI制限が厳しいため、`selenium` と `webdriver_manager` を使用してブラウザ操作で投稿を行うこと。
       - ログイン情報は必ず `.env` から読み込むこと。
    
    2. **要素の特定 (XPath):**
       - Xの投稿ボタンやテキストエリアは `data-testid` 属性を使って特定するのが最も安定する。
       - 例: ツイートボタン -> `//div[@data-testid='tweetButtonInline']`
    
    3. **Google Sheets連携:**
       - `gspread` を使用し、サービスアカウントJSONキーを使って認証する構造にすること。
    
    4. **待機処理:**
       - ページ遷移や投稿完了を待つために `time.sleep` ではなく `WebDriverWait` を使うこと。


## File: `kits\web_article_repurposer.yaml`

yaml
id: "web_article_repurposer"
name: "Web Article to Blog Engine"
description: "Web記事のURLから本文を抽出し、Gemini 2.5 Flashで要約・再構成して、ブログ記事とSNS投稿文を自動生成するStreamlitアプリ。"
version: "1.0.0"

triggers:
  keywords: ["Web要約", "記事要約", "ブログ自動生成", "スクレイピング", "Streamlit"]
  sample_prompts:
    - "このニュース記事をブログに書き直して"
    - "URLから要約とツイートを作るアプリ"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "requests (HTTP Client)"
    - "beautifulsoup4 (HTML Parser)"
    - "google-generativeai (Gemini 1.5 Flash)"
    - "Streamlit (Web UI)"

  core_components:
    - "Scrape Service (Fetch & Parse)"
    - "AI Service (Summarize & Rewrite)"
    - "UI Controller (Streamlit View)"
    - "Main Entry Point"

  expected_file_structure:
    - "app.py"             # UIのエントリーポイント
    - "scrape_service.py"  # 記事抽出ロジック
    - "ai_service.py"      # Gemini APIとの通信ロジック
    - "utils.py"           # 共通ユーティリティ
    - "requirements.txt"

resources:
  domain_knowledge: |
    【開発の鉄則（絶対厳守）】
    1. **メインファイルの役割制限 (Strict Rule):**
       - `app.py` には **UIの表示コード（ボタン、テキストボックスなど）以外を書いてはならない**。
       - スクレイピングやAI生成の実装は、必ず `scrape_service.py` や `ai_service.py` に記述し、インポートして使用すること。
       - **`main.py` (app.py) へのロジックベタ書きは厳禁。**
    
    2. **スクレイピングの安定性:**
       - ニュースサイトなどはアクセス制限が厳しい場合があるため、`User-Agent` ヘッダーを適切に設定してリクエストすること。
       - 本文抽出が難しい場合でもエラーで落ちず、「本文が取得できませんでした」とUIに表示する安全設計にすること。
    
    3. **AIモデル:**
       - 高速かつ安価な **Gemini 2.5 Flash** を使用すること。


## File: `kits\web_flask.yaml`

yaml
id: "web_flask_standard"
name: "Standard Flask Web App"
description: "Python Flaskを使用した堅牢なWebアプリケーション構成"
version: "1.0.0"

triggers:
  keywords: ["web", "flask", "site", "homepage", "アプリ"]
  sample_prompts:
    - "FlaskでToDoアプリを作って"
    - "シンプルなWebサイトを構築したい"

blueprint:
  suggested_tech_stack:
    - "Python 3.10+"
    - "Flask (Web Framework)"
    - "SQLite (Database)"
    - "Bootstrap 5 (CSS Framework)"
  
  core_components:
    - "Application Factory Pattern (create_app)"
    - "Blueprints (ルーティング分割)"
    - "Jinja2 Templates"

  expected_file_structure:
    - "app.py"
    - "requirements.txt"
    - "src/__init__.py"
    - "templates/base.html"

resources:
  domain_knowledge: |
    Flaskアプリでは、グローバル変数を使わず 'Application Factory' パターンを採用してください。
    HTMLは必ず 'templates/base.html' を継承し、重複コードを防ぐこと。
    データベース接続はリクエストごとに確実にクローズすること。


## File: `src\__init__.py`

py



## File: `src\config.py`

py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    # --- API Configuration ---
    LLM_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    
    # --- Model Strategy (全モデルをStandard Flashに統一し、初期品質を最大化) ---
    # Flash-Liteは廃止し、生成から修復まで全てStandard Flashで実行
    LLM_MODEL_FAST: str = "gemini-2.5-flash"
    LLM_MODEL_HEALER: str = "gemini-2.5-flash"
    LLM_MODEL_SMART: str = "gemini-2.5-flash"
    LLM_MODEL_AUDIT: str = "gemini-2.5-flash"
    
    # --- Budget ---
    MAX_BUDGET_PER_RUN: float = 50.0  # 1回の実行上限 (円)
    
    # --- Application Paths ---
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR: str = os.path.join(BASE_DIR, "evo_output")
    KITS_DIR: str = os.path.join(BASE_DIR, "kits")
    
    # --- Runtime Settings ---
    DOCKER_IMAGE: str = "evo-sandbox"
    CONTAINER_PREFIX: str = "evo-dev"
    MAX_RETRIES: int = 1 # 1回勝負に固定

    def __post_init__(self):
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.KITS_DIR, exist_ok=True)

config = Settings()


## File: `src\services\architect_service.py`

py
import json
import re
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("Architect")

class ArchitectService:
    
    def __init__(self, client, kit_manager):
        self.client = client
        self.kit_manager = kit_manager

    # 戻り値の型ヒントをTuple[List[Dict], Optional[Dict]]とする
    def create_plan(self, user_prompt: str) -> Tuple[List[Dict], Optional[Dict]]:
        """
        ユーザーの要望とKitに基づいて実装フェーズ計画を作成し、その計画と使用したKitを返す。
        """
        
        # 1. キットの選択ロジックを統合
        matches = self.kit_manager.find_best_match(user_prompt)
        kit = matches[0][0] if matches else None
        
        kit_info = ""
        if kit:
            logger.info(f"🧩 Kit Auto-Selected: {kit.get('name')}")
            # トークン節約: 必要な情報だけを抽出
            tech_stack = ', '.join(kit.get('blueprint', {}).get('suggested_tech_stack', []))
            core_comps = ', '.join(kit.get('blueprint', {}).get('core_components', []))
            kit_info = f"## Active Kit: {kit.get('name')}\nStack: {tech_stack}\nComponents: {core_comps}\n"

        sys_prompt = f"""
        You are a Software Architect.
        Break down the user's request into logical implementation phases.
        
        {kit_info}
        
        # CRITICAL RULES:
        1. Separation: UI Logic -> app.py (Streamlit allowed). Data Logic -> *_service.py (NO Streamlit).
        2. Simplicity: Create minimum viable files.
        3. Output JSON Array only.
        """
        
        try:
            # Planの生成 (Flash Standardモデルを使用)
            response = self.client.generate(f"Request: {user_prompt}", sys_prompt)
            
            json_str = response.strip()

            # 🚨 修正: 構文エラーを引き起こしていた正規表現ロジックを、文字列操作に置き換え 🚨
            start_index = json_str.find('[')
            last_index = json_str.rfind(']')
            
            if start_index == -1 or last_index == -1 or start_index > last_index:
                 # JSON配列が見つからなかった場合
                 raise ValueError("JSON array boundary ([...]) not found in response.")

            # []で囲まれた部分を抽出
            final_json_data = json_str[start_index : last_index + 1]

            if final_json_data:
                parsed_plan = json.loads(final_json_data)
                
                # ★★★ 修正ロジックの追加 ★★★
                # Planの「files」リストをクリーンアップし、辞書ではなくファイル名（文字列）のみを抽出する
                cleaned_plan = self._clean_plan_files(parsed_plan)
                
                # PlanとKitを両方返す (2要素のタプル)
                return (cleaned_plan, kit) 
            else:
                raise ValueError("Extracted JSON data is empty.")
                
        except Exception as e:
            logger.error(f"Planning failed: {e}. Attempting fallback.")
            # 💡 修正: フォールバックプランを空のリストではなく、app.pyを生成するプランに戻す
            # このプランが壊れていないか、念のため構造チェックを強制する
            fallback_plan = [{"phase": "1", "description": "Implementation", "files": ["app.py"]}]
            
            # フォールバックプラン自体もクリーンアップをかける (二重保証)
            cleaned_fallback = self._clean_plan_files(fallback_plan)

            return (cleaned_fallback, kit) 
            
    def _clean_plan_files(self, plan: List[Dict]) -> List[Dict]:
        """
        LLMが「files」リスト内に辞書を入れてしまった場合、それをファイル名に変換する。
        また、プラン要素が壊れていないか確認する。
        """
        cleaned_plan = []
        for step in plan:

            # --- SUPER PATCH 2: 説明文の揺らぎ吸収 ---
            for k in ['objective', 'summary', 'desc', 'overview', 'goal']:
                if k in step: step['description'] = step.pop(k)
            # ----------------------------------------


            # --- SUPER PATCH: ファイル検出ロジック ---
            # 1. 既知のキーをチェック
            known_keys = ['target_files', 'files_to_modify', 'file_list', 'modified_files', 'files_to_create', 'code_files', 'output_files']
            for k in known_keys:
                if k in step: step['files'] = step.pop(k)

            # 2. それでも無ければ、値の中身を走査して「ファイルっぽいリスト」を自動発見する
            if 'files' not in step:
                for v in step.values():
                    # 文字列のリストで、.py や .txt で終わるものがあれば採用
                    if isinstance(v, list) and v and isinstance(v[0], str) and (v[0].endswith('.py') or v[0].endswith('.txt')):
                        step['files'] = v
                        break
                    # 辞書のリストで、filenameキーを持っていたら採用
                    if isinstance(v, list) and v and isinstance(v[0], dict) and ('filename' in v[0] or 'name' in v[0]):
                        step['files'] = v
                        break
            # ----------------------------------------


            # --- 柔軟性向上パッチ: あらゆるキーの揺らぎを吸収 ---
            for k in ['phase_title', 'phase_name', 'name', 'step_name', 'title']:
                if k in step: step['phase'] = step.pop(k)
            
            for k in ['target_files', 'files_to_modify', 'file_list', 'modified_files']:
                if k in step: step['files'] = step.pop(k)
            # ------------------------------------------------

            # Planの要素が辞書であることを確認
            if not isinstance(step, dict):
                logger.warning(f"Skipping malformed plan step: {step}")
                continue
                
            files = step.get('files', [])
            cleaned_files = []
            
            for f in files:
                if isinstance(f, dict):
                    # 辞書の場合は'filename'キーを探して文字列に変換
                    if 'filename' in f:
                        cleaned_files.append(f['filename'])
                    elif 'name' in f:
                        cleaned_files.append(f['name'])
                elif isinstance(f, str):
                    # 文字列はそのまま採用
                    cleaned_files.append(f)

            # 念のため、'phase'キーが欠落している場合に備え、フォールバックプランで保証されているか確認
            for k in ['phase_title', 'phase_name', 'name', 'step_name']:
                if k in step: step['phase'] = step.pop(k)
            if 'phase' not in step or 'description' not in step:
                 logger.warning(f"Plan step missing required keys: {step}")
                 # 壊れたステップをスキップする
                 continue

            step['files'] = cleaned_files
            cleaned_plan.append(step)
            
        return cleaned_plan


## File: `src\services\budget_service.py`

py
import logging

logger = logging.getLogger("BudgetGuard")

class BudgetGuard:
    def __init__(self, limit_yen=50.0):
        self.limit_yen = limit_yen
        self.current_cost = 0.0
        
        # 100万トークンあたりの単価目安 (円) - $1=150円換算
        # 参考: https://ai.google.dev/gemini-api/docs/pricing
        
        # Flash / Flash-Lite: 
        # Input $0.075 (~11.25円) / Output $0.30 (~45円) 
        # ※以前の画像($0.10/$0.40)より安くなっていますが、安全側に倒して少し高めに設定するか、正確に合わせるか。
        # ここでは安全マージン込みで $0.10 / $0.40 (15円 / 60円) を維持しつつ、Proを修正します。
        
        # Pro: 
        # Input $1.25 (~187.5円) / Output $10.00 (~1500円)
        # ※<=128kコンテキストの場合。これを超えると倍額になりますが、基本はこちらを使用。
        
        self.rates = {
            # Flash系 (Lite含む)
            "gemini-2.5-flash-lite": {"input": 15.0,  "output": 60.0},
            "gemini-2.0-flash":      {"input": 15.0,  "output": 60.0},
            "gemini-2.5-flash":      {"input": 15.0,  "output": 60.0}, 
            "gemini-1.5-flash":      {"input": 15.0,  "output": 60.0},
            
            # Pro / High-Intelligence系 (1.5, 2.0, 2.5, 3.0)
            "gemini-2.5-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-2.0-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-1.5-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-3":              {"input": 300.0, "output": 1800.0}, # Gemini 3は仮の高め設定
        }

    def check_and_record(self, model_name: str, input_chars: int, output_chars: int):
        """コストを計算し、累積する。上限を超えたら例外を投げる。"""
        rate = None
        # 部分一致でレートを探す (例: "models/gemini-1.5-pro-latest" -> "gemini-1.5-pro")
        for key in self.rates:
            if key in model_name:
                rate = self.rates[key]
                break
        
        if not rate:
            # 安全策: "pro" が名前に含まれていたらPro価格、それ以外はFlash価格を適用
            if "pro" in model_name.lower():
                rate = self.rates["gemini-1.5-pro"]
            else:
                rate = self.rates["gemini-2.5-flash-lite"]

        input_cost = (input_chars / 1_000_000) * rate["input"]
        output_cost = (output_chars / 1_000_000) * rate["output"]
        total_cost = input_cost + output_cost
        
        self.current_cost += total_cost
        
        logger.info(f"💰 Cost: +{total_cost:.4f}円 (Total: {self.current_cost:.2f} / {self.limit_yen}円) [{model_name}]")

        if self.current_cost > self.limit_yen:
            logger.error("💸 BUDGET EXCEEDED! Stopping execution to save money.")
            raise Exception(f"Budget Limit Exceeded: Used {self.current_cost:.2f}JPY (Limit: {self.limit_yen}JPY)")


## File: `src\services\data_recorder.py`

py
import json
import os
import logging
from datetime import datetime
from src.config import config

logger = logging.getLogger("DataRecorder")

class DataRecorder:
    def __init__(self):
        self.data_dir = os.path.join(config.BASE_DIR, "datasets")
        os.makedirs(self.data_dir, exist_ok=True)
        self.dataset_path = os.path.join(self.data_dir, "evo_success_log.jsonl")

    def save_success(self, prompt: str, kit_name: str, final_files: dict):
        """
        成功体験をデータセットに追加する
        Format: Alpaca / Llama 3 Instruction Tuning Format
        """
        try:
            # 必要なコードファイルだけを抽出
            code_content = ""
            for fname, content in final_files.items():
                if fname.endswith(('.py', '.js', '.html', '.css')):
                    code_content += f"# File: {fname}\n{content}\n\n"

            entry = {
                "timestamp": datetime.now().isoformat(),
                "instruction": prompt,
                "input": f"Use Kit: {kit_name}" if kit_name else "No Kit",
                "output": code_content,
                "system": "You are Evo, an expert AI developer."
            }

            # JSONL形式（1行1JSON）で追記
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"💾 Success data recorded to {self.dataset_path}")

        except Exception as e:
            logger.error(f"Failed to record data: {e}")


## File: `src\services\git_service.py`

py
import subprocess
import os
import logging
from src.config import config

logger = logging.getLogger("Git")

class GitService:
    def __init__(self):
        self.cwd = config.OUTPUT_DIR
        self._init_repo()

    def _init_repo(self):
        if not os.path.exists(os.path.join(self.cwd, ".git")):
            try:
                self._run(["init"])
                # .gitignore作成
                with open(os.path.join(self.cwd, ".gitignore"), "w") as f:
                    f.write(".venv/\n__pycache__/\n_trash/\n*.log\n")
                self._run(["add", "."])
                self._run(["commit", "-m", "Init"])
            except: pass

    def commit(self, msg):
        try:
            self._run(["add", "."])
            # 変更がある場合のみコミット
            if self._run(["status", "--porcelain"], capture_output=True):
                self._run(["commit", "-m", msg])
                logger.info(f"🕰️ Git saved: {msg}")
        except: pass

    # ★追加: 現在のコミットハッシュを取得 (ロールバック用)
    def get_head_hash(self):
        try: return self._run(["rev-parse", "HEAD"], capture_output=True).strip()
        except: return None

    # ★追加: 指定したコミットまで強制的に巻き戻す
    def revert_to(self, commit_hash):
        if not commit_hash: return
        try:
            self._run(["reset", "--hard", commit_hash])
            logger.warning(f"⏪ Reverted code to snapshot: {commit_hash[:7]}")
        except Exception as e:
            logger.error(f"Revert failed: {e}")

    def _run(self, args, capture_output=False):
        return subprocess.run(
            ["git"] + args, 
            cwd=self.cwd, 
            check=True, 
            capture_output=capture_output, 
            text=True, 
            encoding='utf-8'
        ).stdout


## File: `src\services\healer_service.py`

py
import logging
import hashlib
from typing import Dict, List, Tuple, Optional
from src.services.patch_service import PatchService

logger = logging.getLogger("Healer")

class HealerService:
    def __init__(self, fast_client, healer_client):
        self.fast = fast_client     # L1/L2
        self.healer = healer_client # L3 (Flash Standard)
        self.patcher = PatchService()
        self.repair_history = {} 

    def build_context(self, files: Dict[str, str]) -> str:
        # コンテキストサイズ削減: 先頭1000文字だけ渡す
        context = []
        for name, content in files.items():
            snippet = content[:1000] + "\n...(truncated)..." if len(content) > 1000 else content
            context.append(f"File: {name}\n```\n{snippet}\n```")
        return "\n".join(context)

    def heal(self, fname: str, content: str, errors: List[str], context_files: Dict, kit: Optional[Dict] = None) -> Tuple[bool, str, str]:
        error_msg = errors[0] if errors else "Unknown error"
        
        # --- ループ検知ロジック ---
        error_hash = hashlib.md5(error_msg.encode('utf-8')).hexdigest()
        history_key = f"{fname}:{error_hash}"
        current_tries = self.repair_history.get(history_key, 0)
        
        if current_tries >= 2: # 2回試してダメなら諦める
            logger.warning(f"🛑 Healing Loop Detected for {fname}. Ignoring error and proceeding.")
            # ★重要: FalseではなくTrueを返し、変更なしのコンテンツを返すことでプロセスを止めない
            return True, content, "Loop_Ignored"
        
        self.repair_history[history_key] = current_tries + 1

        context_str = self.build_context(context_files)
        kit_instruction = ""
        if kit:
            kit_instruction = f"Context: {kit.get('name')}"

        base_prompt = f"""
        Fix code in '{fname}'.
        Error: {error_msg}
        {kit_instruction}
        
        Current Code:
        {content}
        
        Reference:
        {context_str}
        """

        # L2: Patch (安価)
        try:
            prompt_l2 = base_prompt + "\nReturn a SEARCH/REPLACE block (<<<< SEARCH ... ==== ... >>>>)."
            patch_res = self.fast.generate(prompt_l2)
            patched_code = self.patcher.apply_patch(content, patch_res)
            if patched_code: return True, patched_code, "L2_Patch"
        except Exception: pass

        # L3: Rewrite (高価だが確実) - ループ1回目の時だけ試す
        if current_tries == 0:
            try:
                prompt_l3 = base_prompt + "\nRewrite the FULL file correctly. Output only the code."
                fixed_res = self.healer.generate(prompt_l3)
                fixed_code = self._clean_code(fixed_res)
                if len(fixed_code) > 10: return True, fixed_code, "L3_Rewrite"
            except Exception as e:
                logger.error(f"Healer failed: {e}")

        # 修正できなくても、ロールバックさせないために元のコードを返す
        logger.warning(f"⚠️ Could not fix {fname}. Keeping original.")
        return True, content, "Skipped"

    def _clean_code(self, text):
        return text.replace("```python", "").replace("```", "").strip()


## File: `src\services\kit_gen_service.py`

py
import logging
from typing import Dict, Optional

logger = logging.getLogger("KitGenService")

class KitGenService:
    """
    ユーザーの要望に基づいて、Evo自身の拡張プラグイン(Kit YAML)を生成するサービス。
    自己進化の中核を担う。
    """
    def __init__(self, client):
        self.client = client # Smart Client (Pro/Flash) を使用

    def generate_kit(self, user_prompt: str) -> str:
        """
        ユーザーの自然言語記述から、有効なKit YAMLを生成する
        """
        system_prompt = """
        あなたはAIエージェント「Evo」の機能拡張エンジニアです。
        ユーザーの要望に基づき、Evoが特定のタスクを遂行するための「Kit（専門知識定義ファイル）」をYAML形式で作成してください。

        【Kitの構成要素】
        1. id: 一意の識別子 (英数字とアンダースコア)
        2. name: わかりやすい名前
        3. description: 何をするKitか
        4. triggers: このKitが発動すべきキーワードとサンプルプロンプト
        5. blueprint: 推奨技術スタック、主要コンポーネント、ファイル構成
        6. resources: **最重要**。AIに与えるドメイン知識、設計思想、ベストプラクティス。

        【出力ルール】
        - 必ず有効なYAML形式のみを出力すること。
        - マークダウンのコードブロック (```yaml ... ```) で囲むこと。
        - `domain_knowledge` は具体的かつ専門的に書くこと（ライブラリの正しい使い方、落とし穴、設計パターンなど）。

        【出力例】
        ```yaml
        id: "discord_bot_py"
        name: "Discord Bot Builder"
        description: "discord.pyを使用した高機能Bot開発キット"
        triggers:
          keywords: ["discord", "bot", "ディスコード"]
          sample_prompts: ["サーバー管理Botを作って"]
        blueprint:
          suggested_tech_stack: ["Python 3.10", "discord.py", "python-dotenv"]
          core_components: ["Event Listener", "Command Tree", "Cog System"]
          expected_file_structure:
            - "main.py"
            - "cogs/general.py"
            - ".env"
        resources:
          domain_knowledge: |
            discord.py 2.0以降では `Intents` の設定が必須です。
            大規模なBotの場合は `Cogs` 機能を使ってコマンドを分割管理してください。
            トークンは必ず環境変数から読み込むこと。
        ```
        """

        prompt = f"以下の要望を満たすKitを作成してください:\n{user_prompt}"
        
        logger.info("🧠 Generating new Kit definition...")
        response = self.client.generate(prompt, system_prompt)
        
        # クリーニング (Markdown除去)
        yaml_content = response.replace("```yaml", "").replace("```", "").strip()
        return yaml_content


## File: `src\services\kit_manager.py`

py
import os
import yaml
import glob
import logging
from typing import Dict, List, Tuple
from src.config import config

logger = logging.getLogger("KitManager")

class KitManager:
    def __init__(self, client=None):
        self.kits = {}
        # Clientは受け取るが、基本使わない（コスト削減）
        self.client = client
        self._load_all_kits()
    
    def _load_all_kits(self):
        self.kits = {}
        if not os.path.exists(config.KITS_DIR): return
        
        for f in glob.glob(os.path.join(config.KITS_DIR, "*.yaml")):
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = yaml.safe_load(fp)
                    if data and 'id' in data:
                        self.kits[data['id']] = data
            except Exception as e:
                logger.error(f"Error loading kit {f}: {e}")
        
        logger.info(f"📦 Kits Loaded: {len(self.kits)}")

    def find_best_match(self, prompt: str, top_n=1) -> List[Tuple[Dict, float]]:
        """
        キーワードマッチのみを使用し、LLMコストをゼロにする。
        """
        matches = []
        p_lower = prompt.lower()
        
        for kit_id, kit in self.kits.items():
            score = 0
            # キーワード一致
            for kw in kit.get('triggers', {}).get('keywords', []):
                if kw.lower() in p_lower: 
                    score += 5.0 # キーワードヒットは重みを大きく
            
            # 説明文の部分一致（簡易的）
            if kit.get('description', '').lower() in p_lower:
                score += 2.0

            if score > 0:
                matches.append((kit, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        
        if matches:
            logger.info(f"⚡ Kit matched by keyword: {matches[0][0]['name']}")
            return matches[:top_n]
        
        # マッチしなかった場合、AIに聞くロジックを入れることもできるが、
        # コスト優先なら「キットなし」で進めるのが正解。
        return []

    def save_new_kit(self, yaml_content: str) -> str:
        try:
            data = yaml.safe_load(yaml_content)
            if not data or 'id' not in data: raise ValueError("Invalid YAML")
            path = os.path.join(config.KITS_DIR, f"{data['id']}.yaml")
            with open(path, 'w', encoding='utf-8') as f:
                f.write(yaml_content)
            self._load_all_kits()
            return data.get('name', data['id'])
        except Exception as e:
            logger.error(f"Failed to save kit: {e}")
            raise e


## File: `src\services\patch_service.py`

py
import re
import logging
from typing import Optional

logger = logging.getLogger("EvoPatchService")

class PatchService:
    @staticmethod
    def normalize(code: str) -> str:
        code = re.sub(r'#.*', '', code)
        return re.sub(r'\s+', ' ', code).strip()

    def apply_patch(self, original_code: str, patch_text: str) -> Optional[str]:
        pattern = re.compile(r"<<<< SEARCH\n(.*?)\n====\n(.*?)\n>>>>", re.DOTALL)
        matches = pattern.findall(patch_text)
        if not matches: return None
        
        new_code = original_code
        
        for search_block, replace_block in matches:
            if original_code.count(search_block) > 1:
                logger.warning("Patch Skipped: Non-unique search block.")
                return None
            if search_block in new_code:
                new_code = new_code.replace(search_block, replace_block, 1)
                continue
            
            # Fuzzy match fallback
            search_norm = self.normalize(search_block)
            lines = new_code.split('\n')
            n_search = len(search_block.split('\n'))
            
            for i in range(len(lines) - n_search + 1):
                candidate_block = "\n".join(lines[i:i+n_search])
                if self.normalize(candidate_block) == search_norm:
                    lines[i:i+n_search] = replace_block.split('\n')
                    new_code = "\n".join(lines)
                    break
        return new_code if new_code != original_code else None


## File: `src\services\qa_service.py`

py
import logging

logger = logging.getLogger("EvoQA")

class QualityAssuranceService:
    def __init__(self, client):
        self.client = client # Healerと同じ賢いモデル(Flash)推奨

    def audit_and_fix(self, project_files: dict) -> str:
        """
        最終監査: ファイル間の不整合（Importミス、関数引数不一致）のみをチェックする。
        ロジックの中身までは見ないことでコストを削減。
        """
        context_str = self._build_lightweight_context(project_files)
        if not context_str: return ""
        
        system_prompt = """
        Role: QA Engineer.
        Task: Check consistency between files. Ignore logic bugs inside functions.
        
        Focus on:
        1. **Import Errors**: Does the imported function exist in the target file?
        2. **Signature Mismatch**: Do function calls match definitions?
        3. **HTML/JS IDs**: Do JS `getElementById` IDs match HTML `id`s?

        Output Format:
        If you find a CRITICAL integration bug, output the FULL fixed file content:
        # FILENAME: path/to/file.py
        ```python
        ... code ...
        ```
        If everything looks consistent, output NOTHING.
        """
        
        user_prompt = f"Audit these file interfaces:\n\n{context_str}"

        try:
            # 賢いモデルで一発で決める
            return self.client.generate(user_prompt, system_prompt)
        except Exception as e:
            logger.error(f"Audit failed: {e}")
            return ""

    def _build_lightweight_context(self, project_files):
        # コンテキストサイズ削減: 
        # コードの中身を全部渡すのではなく、構造を渡すべきだが、
        # 修正させるためにはコードが必要。
        # 妥協案: 主要なコードファイルのみ渡し、巨大なデータファイルやConfigは除外する。
        
        valid_exts = {'.py', '.js', '.html'}
        content = []
        
        for fname, code in project_files.items():
            if any(fname.endswith(ext) for ext in valid_exts):
                # 2000行を超えるような巨大ファイルは、先頭と末尾だけ渡す等の工夫も可能だが、
                # ここでは単純に文字数制限を設ける
                if len(code) > 20000: 
                    snippet = code[:5000] + "\n... (truncated for QA) ...\n" + code[-5000:]
                else:
                    snippet = code
                content.append(f"# FILENAME: {fname}\n```\n{snippet}\n```")
        
        return "\n".join(content)


## File: `src\services\search_service.py`

py
import logging
from ddgs import DDGS # パッケージ名変更に対応
from src.config import config

logger = logging.getLogger("SearchService")

class SearchService:
    """
    Web検索サービス (Cost Optimized)
    ページごとの要約(N回)をやめ、スニペット集約→最終回答(1回)に変更。
    """
    def __init__(self, client):
        self.client = client # Flash-Lite
        self.ddgs = DDGS()

    def research(self, query: str, max_results=3) -> str:
        logger.info(f"🔍 Searching for: '{query}'...")
        
        try:
            # 1. DuckDuckGoで検索 (無料)
            # bodyキーにスニペットが入っているのでこれを使う
            results = list(self.ddgs.text(query, max_results=max_results))
            if not results:
                return "No search results found."
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search Error: {str(e)}"

        # 2. コンテキストの集約 (ページにはアクセスしない)
        # 実際にページを開いてスクレイピングするのは時間がかかり、
        # スクレイピング対策で失敗することも多いため、検索エンジンの要約を信じる。
        
        context_data = ""
        for i, r in enumerate(results):
            title = r.get('title', 'No Title')
            link = r.get('href', '')
            snippet = r.get('body', '')
            context_data += f"Source {i+1}: {title}\nURL: {link}\nSummary: {snippet}\n\n"

        # 3. 1回だけLLMを呼び出してレポート作成
        prompt = f"""
        User Query: "{query}"

        Search Results:
        {context_data}

        Task: Summarize the search results to answer the user's query.
        Focus on technical details (libraries, code usage, installation).
        Output Format: Markdown
        """
        
        try:
            report = self.client.generate(prompt, "Role: Tech Researcher. Output: Concise technical summary.")
            return report
        except Exception as e:
            return f"Failed to generate report: {e}"


## File: `src\services\structure_service.py`

py
import os
import ast
import json
import logging
from typing import Dict, List

logger = logging.getLogger("StructureService")

class StructureService:
    def __init__(self):
        self.dependency_graph = {}
        self.symbol_table = {}

    def analyze_project(self, files: Dict[str, str]) -> str:
        """
        プロジェクト全体を解析し、依存関係と定義済みシンボル（関数・クラス）のマップを作成する。
        これをLLMのコンテキストに注入することで、全体構造を理解させる。
        """
        self.dependency_graph = {}
        self.symbol_table = {}

        for fname, content in files.items():
            if fname.endswith('.py'):
                self._analyze_python_file(fname, content)
        
        # LLMに渡すための要約テキストを生成
        summary = "# Project Structure Summary\n"
        
        summary += "## Defined Symbols (Classes & Functions):\n"
        for fname, symbols in self.symbol_table.items():
            # シンボルがないファイルはスキップしてトークン節約
            if not symbols: continue
            
            summary += f"- **{fname}**:\n"
            for sym in symbols:
                summary += f"  - `{sym['type']} {sym['name']}` (Line {sym['line']})\n"
        
        summary += "\n## Dependencies (Imports):\n"
        for fname, deps in self.dependency_graph.items():
            if deps:
                summary += f"- **{fname}** depends on: {', '.join(deps)}\n"
                
        return summary

    def _analyze_python_file(self, fname: str, code: str):
        try:
            tree = ast.parse(code)
            symbols = []
            imports = []

            for node in ast.walk(tree):
                # クラス定義
                if isinstance(node, ast.ClassDef):
                    # トークン節約: プライベートクラス（_で始まる）は地図に載せない
                    if node.name.startswith('_'): continue 
                    symbols.append({'type': 'class', 'name': node.name, 'line': node.lineno})
                
                # 関数定義
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # トークン節約: プライベート関数（_で始まる）は地図に載せない
                    if node.name.startswith('_'): continue
                    symbols.append({'type': 'function', 'name': node.name, 'line': node.lineno})
                
                # インポート
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for n in node.names: imports.append(n.name.split('.')[0])
                    elif node.module:
                        imports.append(node.module.split('.')[0])

            self.symbol_table[fname] = symbols
            self.dependency_graph[fname] = list(set(imports)) # 重複排除

        except Exception as e:
            logger.warning(f"Failed to parse {fname}: {e}")


## File: `src\services\verifier_service.py`

py
import ast
import logging
import autopep8
import os
import re
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Set

logger = logging.getLogger("Verifier")

class VerifierService:
    def __init__(self, runtime):
        self.runtime = runtime
        # 危険な操作のみ禁止。ライブラリの使用制限は撤廃（コスト削減のため）
        self.BANNED_MODULES = ['subprocess', 'socket']
        self.BANNED_FUNCTIONS = ['eval', 'exec'] 

    def format_code(self, code: str, filename: str) -> str:
        try:
            if filename.endswith(".py"):
                return autopep8.fix_code(code, options={'aggressive': 1})
            if filename.endswith(('.html', '.js', '.css', '.json', '.yaml', '.md')):
                return code.strip() + "\n"
        except: pass
        return code

    def verify(self, code: str, filename: str, context_files: dict = None) -> dict:
        """
        静的解析: 致命的な構文エラーのみをチェックする (過剰品質の排除)
        """
        ext = os.path.splitext(filename)[1].lower()
        errors = []

        # 1. Pythonの検査
        if ext == '.py':
            try: tree = ast.parse(code)
            except SyntaxError as e: return {"valid": False, "errors": [f"Python Syntax: {e}"]}
            
            # セキュリティチェックのみ実施
            sec = self._check_banned_nodes(tree)
            if not sec['valid']: errors.extend(sec['errors'])
            
            # アーキテクチャチェック(_check_architecture)は廃止
            # 理由: AIが混乱し、修正ループに陥る最大の原因であるため。
            
            # コンテキスト整合性チェック (Importエラーのみ確認)
            if context_files:
                symbol_table = self._build_symbol_table(context_files)
                import_errors = self._verify_imports(tree, filename, symbol_table)
                errors.extend(import_errors)

        # 2. JSON, HTML検査
        elif ext == '.json':
            try: json.loads(code)
            except Exception as e: errors.append(f"JSON Error: {e}")
        elif ext == '.html':
            if '<body>' not in code and '<body ' not in code: errors.append("Missing <body> tag")

        return {"valid": len(errors) == 0, "errors": errors}

    # 重複チェックなどの過剰機能は削除し、単純化
    
    def _build_symbol_table(self, files: Dict[str, str]) -> Dict[str, Set[str]]:
        symbols = {}
        for fname, content in files.items():
            if not fname.endswith('.py'): continue
            module_name = os.path.splitext(os.path.basename(fname))[0]
            defined = set()
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        defined.add(node.name)
                symbols[module_name] = defined
                symbols[fname] = defined 
            except: pass
        return symbols

    def _verify_imports(self, tree: ast.AST, current_filename: str, symbol_table: Dict[str, Set[str]]) -> List[str]:
        errors = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
                # 外部ライブラリはスキップ
                if module_name not in symbol_table and module_name + ".py" not in symbol_table:
                    continue
                
                defined_symbols = symbol_table.get(module_name, set())
                if not defined_symbols: defined_symbols = symbol_table.get(module_name + ".py", set())

                for alias in node.names:
                    if alias.name == '*': continue
                    if alias.name not in defined_symbols:
                        # 致命的ではないが、警告として記録
                        errors.append(f"Import Warning: '{alias.name}' not found in '{module_name}'.")
        return errors

    def _check_banned_nodes(self, tree):
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split('.')[0] in self.BANNED_MODULES: return {"valid":False, "errors":[f"Banned import: {a.name}"]}
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split('.')[0] in self.BANNED_MODULES: return {"valid":False, "errors":[f"Banned import: {node.module}"]}
            elif isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name): func_name = node.func.id
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id in self.BANNED_MODULES: return {"valid":False, "errors":[f"Banned call: {node.func.value.id}"]}
                if func_name in self.BANNED_FUNCTIONS: return {"valid":False, "errors":[f"Banned function: {func_name}"]}
        return {"valid": True, "errors": []}


## File: `src\services\workspace_manager.py`

py
import os
import re
import logging
from typing import Dict
from src.config import config
from src.services.git_service import GitService

logger = logging.getLogger("Workspace")

class WorkspaceManager:
    """
    ファイル操作、Git、コードのパースなど、
    「思考」以外の「作業」を一手に引き受けるクラス。
    """
    def __init__(self):
        self.project_files: Dict[str, str] = {}
        self.git = GitService()
        self._load_workspace()

    def _load_workspace(self):
        self.project_files = {}
        ignore = {'.venv', '__pycache__', '_trash', '.git', 'node_modules'}
        for root, dirs, files in os.walk(config.OUTPUT_DIR):
            dirs[:] = [d for d in dirs if d not in ignore]
            for file in files:
                if file.endswith(('.py', '.html', '.js', '.css', '.json', '.md', '.txt', '.yaml')):
                    try:
                        path = os.path.join(root, file)
                        rel_path = os.path.relpath(path, config.OUTPUT_DIR).replace("\\", "/")
                        with open(path, 'r', encoding='utf-8') as f:
                            self.project_files[rel_path] = f.read()
                    except: pass

    def save_file(self, fname: str, content: str):
        # コードブロックの除去などをここで統一して行う
        content = self._clean_code(content)
        
        path = os.path.abspath(os.path.join(config.OUTPUT_DIR, fname))
        if not path.startswith(os.path.abspath(config.OUTPUT_DIR)): return # パス漏洩防止
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        self.project_files[fname] = content

    def parse_and_save_files(self, llm_response: str, default_filename: str = None) -> Dict[str, str]:
        """LLMの出力からファイルを抽出して保存する"""
        files = {}
        # # FILENAME: ... パターン
        pattern = re.compile(r"^#\s*FILENAME:\s*(?P<name>[^\n]+)\n(?P<code>.*?)(?=^#\s*FILENAME:|\Z)", re.DOTALL | re.MULTILINE)
        matches = list(pattern.finditer(llm_response))
        
        if matches:
            for match in matches:
                fname = match.group("name").strip()
                code = match.group("code").strip()
                files[fname] = code
                self.save_file(fname, code)
        elif default_filename:
            # パターンがない場合は全体を一つのファイルとして扱う
            self.save_file(default_filename, llm_response)
            files[default_filename] = llm_response
            
        return files

    def _clean_code(self, text: str) -> str:
        # マークダウン記法の除去
        return text.replace("```python", "").replace("```json", "").replace("```", "").strip()

    def add_to_requirements(self, pkg: str):
        path = "requirements.txt"
        current = self.project_files.get(path, "")
        if pkg not in current:
            new_content = current.strip() + f"\n{pkg}\n"
            self.save_file(path, new_content)

    def commit(self, message: str):
        self.git.commit(message)


## File: `templates\index.html`

html
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evo Studio v2.6 (Retro IDE)</title>
    
    <!-- CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vue/3.3.4/vue.global.min.js"></script>
    <!-- フォント: DotGothic16 (日本語) & Press Start 2P (英数) -->
    <link href="https://fonts.googleapis.com/css2?family=DotGothic16&family=Press+Start+2P&display=swap" rel="stylesheet">
    <!-- FontAwesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    
    <!-- Highlight.js -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/atom-one-dark.min.css">

    <style>
        /* カラーパレット (落ち着いたレトロダーク) */
        :root {
            --bg-main: #282c34;
            --bg-panel: #21252b;
            --bg-header: #181a1f;
            --accent-primary: #98c379; /* レトログリーン */
            --accent-secondary: #61afef; /* レトロブルー */
            --text-main: #abb2bf;
            --text-highlight: #ffffff;
            --border-color: #3e4451;
        }

        body { 
            /* 日本語は DotGothic16, 英数字は Press Start 2P */
            font-family: 'Press Start 2P', 'DotGothic16', sans-serif;
            background-color: var(--bg-main);
            color: var(--text-main);
            font-size: 12px;
            line-height: 1.5;
            overflow: hidden;
        }

        .jp-font {
            font-family: 'DotGothic16', sans-serif;
            font-weight: bold;
        }

        /* UIパーツ */
        .retro-btn {
            background-color: var(--bg-panel);
            color: var(--accent-secondary);
            border: 1px solid var(--border-color);
            transition: all 0.1s;
            font-family: 'DotGothic16', sans-serif;
            cursor: pointer;
        }
        .retro-btn:hover {
            background-color: var(--border-color);
            color: var(--text-highlight);
        }
        .retro-btn:active {
            transform: translateY(2px);
        }
        .retro-btn.primary {
            background-color: var(--accent-primary);
            color: #1e2227;
            border: none;
        }
        .retro-btn.primary:hover {
            opacity: 0.9;
        }

        .retro-input {
            background-color: #1e2227;
            border: 1px solid var(--border-color);
            color: var(--text-highlight);
            font-family: 'DotGothic16', sans-serif;
            font-size: 14px;
        }
        .retro-input:focus {
            outline: none;
            border-color: var(--accent-secondary);
        }

        /* スクロールバー */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--bg-main); }
        ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #5c6370; }

        /* ファイルリスト */
        .file-item {
            border-bottom: 1px solid var(--bg-header);
            transition: background 0.1s;
        }
        .file-item:hover { 
            background-color: var(--bg-header); 
            color: var(--text-highlight);
            cursor: pointer; 
        }
        .file-item.active { 
            background-color: #323844; 
            color: var(--accent-secondary); 
            border-left: 3px solid var(--accent-secondary);
        }

        /* ハイライト調整 */
        pre code.hljs {
            font-family: 'Consolas', 'Monaco', monospace; /* コードは見やすさ重視 */
            font-size: 14px;
            line-height: 1.6;
            background: transparent;
        }
    </style>
</head>
<body class="h-screen flex flex-col">

    {% raw %}
    <div id="app" class="h-full flex flex-col">
        
        <!-- ヘッダー -->
        <header class="h-12 border-b border-gray-700 flex items-center px-4 justify-between shrink-0 bg-[#181a1f]">
            <div class="flex items-center gap-3">
                <i class="fa-solid fa-terminal text-[#98c379] text-xl"></i>
                <div class="flex flex-col">
                    <span class="font-bold text-[14px] tracking-wide text-white jp-font leading-tight">Evo Studio <span class="text-[10px] text-[#61afef]">v2.6</span></span>
                    <span class="text-[8px] text-gray-500">AI AGENT IDE</span>
                </div>
            </div>
            <div class="flex items-center gap-3">
                <div v-if="activeKit" class="text-[10px] px-2 py-1 bg-[#323844] text-[#98c379] border border-gray-600 rounded flex items-center gap-2 jp-font">
                    <i class="fa-solid fa-microchip"></i> {{ activeKit }}
                </div>
                <a href="/download" class="text-[11px] px-3 py-1 retro-btn rounded no-underline flex items-center gap-2 jp-font">
                    <i class="fa-solid fa-download"></i> 保存
                </a>
            </div>
        </header>

        <!-- メインワークスペース -->
        <div class="flex-1 flex overflow-hidden">
            
            <!-- 左サイドバー: ファイル -->
            <div class="w-64 border-r border-gray-700 flex flex-col shrink-0 bg-[#21252b]">
                <div class="p-3 text-[11px] font-bold text-gray-400 border-b border-gray-700 flex justify-between items-center jp-font">
                    <span><i class="fa-regular fa-folder-open mr-1"></i> プロジェクト</span>
                    <button @click="refreshFiles" class="hover:text-white transition"><i class="fa-solid fa-sync"></i></button>
                </div>
                <div class="flex-1 overflow-y-auto">
                    <div v-for="file in files" :key="file" 
                         @click="loadFile(file)"
                         class="file-item px-4 py-2 text-[12px] flex items-center gap-2"
                         :class="{ 'active': currentFile === file }">
                        <i class="fa-regular fa-file-code text-gray-500"></i> <span class="truncate">{{ file }}</span>
                    </div>
                </div>
                
                <!-- ステータスパネル -->
                <div class="p-3 border-t border-gray-700 bg-[#1e2227]">
                    <div class="text-[10px] font-bold text-gray-500 mb-2 jp-font">ステータス</div>
                    <div class="grid grid-cols-2 gap-2 text-[10px] jp-font">
                        <div class="bg-[#282c34] p-1 rounded border border-gray-700 text-center">
                            <div class="text-[#61afef]">フェーズ</div>
                            <div class="text-white">{{ stats.currentPhase }}/{{ stats.totalPhases }}</div>
                        </div>
                        <div class="bg-[#282c34] p-1 rounded border border-gray-700 text-center">
                            <div class="text-[#98c379]">修復</div>
                            <div class="text-white">{{ stats.l1 + stats.l2 }}</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- 中央: エディタエリア -->
            <div class="flex-1 flex flex-col min-w-0 bg-[#282c34]">
                <!-- タブバー -->
                <div class="h-10 flex border-b border-gray-700 bg-[#21252b]">
                    <button @click="viewMode='code'" class="px-5 text-[11px] flex items-center gap-2 jp-font border-r border-gray-700 transition"
                         :class="viewMode==='code' ? 'bg-[#282c34] text-[#61afef] border-t-2 border-t-[#61afef]' : 'text-gray-500 hover:text-white hover:bg-[#2c313a]'">
                        <i class="fa-solid fa-code"></i> コード
                    </button>
                    <button @click="viewMode='preview'" class="px-5 text-[11px] flex items-center gap-2 jp-font border-r border-gray-700 transition"
                         :class="viewMode==='preview' ? 'bg-[#282c34] text-[#61afef] border-t-2 border-t-[#61afef]' : 'text-gray-500 hover:text-white hover:bg-[#2c313a]'">
                        <i class="fa-solid fa-play"></i> プレビュー
                    </button>
                    <div class="flex-1"></div>
                    <span v-if="currentFile" class="px-4 text-[11px] text-gray-500 self-center font-mono">{{ currentFile }}</span>
                </div>

                <!-- コンテンツ表示 -->
                <div class="flex-1 relative overflow-hidden">
                    <div v-show="viewMode==='code'" class="absolute inset-0 overflow-auto">
                        <pre><code class="language-python h-full" ref="codeBlock">{{ fileContent }}</code></pre>
                    </div>
                    <div v-show="viewMode==='preview'" class="absolute inset-0 bg-white">
                        <iframe v-if="previewUrl" :src="previewUrl" class="w-full h-full border-none"></iframe>
                        <div v-else class="flex flex-col items-center justify-center h-full text-[#333]">
                            <i class="fa-solid fa-eye-slash text-4xl mb-4 text-gray-300"></i>
                            <span class="text-[12px] font-bold text-gray-400 jp-font">プレビューできません</span>
                        </div>
                    </div>
                </div>

                <!-- 下部: コンソール/チャット -->
                <div class="h-1/3 min-h-[150px] border-t border-gray-700 flex flex-col bg-[#21252b]">
                    <div class="h-8 px-4 flex items-center justify-between bg-[#1b1d23] border-b border-gray-700">
                        <span class="text-[11px] font-bold text-[#98c379] jp-font"><i class="fa-solid fa-terminal mr-2"></i>システムログ / チャット</span>
                        <button @click="reset" class="text-[10px] text-gray-500 hover:text-[#e06c75] jp-font"><i class="fa-solid fa-trash mr-1"></i>ログ消去</button>
                    </div>
                    <div class="flex-1 overflow-y-auto p-4 font-mono text-[12px] space-y-3 bg-[#282c34]" ref="chatLog">
                        <div v-for="(msg, i) in messages" :key="i" class="flex gap-3">
                            <div class="font-bold shrink-0 w-12 text-right" :class="{'text-[#61afef]': msg.role==='user', 'text-[#98c379]': msg.role!=='user'}">
                                {{ msg.role === 'user' ? 'YOU' : 'EVO' }}
                            </div>
                            <div class="text-[#abb2bf] flex-1 jp-font leading-relaxed border-l-2 border-gray-700 pl-3" v-html="formatMessage(msg.content)"></div>
                        </div>
                    </div>
                    
                    <!-- 入力エリア -->
                    <div class="p-3 border-t border-gray-700 flex gap-3 bg-[#21252b]">
                        <div class="relative flex-1">
                            <i class="fa-solid fa-chevron-right absolute left-3 top-3 text-gray-500 text-xs"></i>
                            <input v-model="prompt" @keydown.enter="generate" 
                                class="w-full retro-input pl-8 pr-4 py-2.5 rounded shadow-inner"
                                placeholder="Evoへの指示を入力してください... (例: MITライセンス紹介ブログを作って)">
                        </div>
                        <button @click="generate" :disabled="loading" class="retro-btn primary px-6 py-2 rounded font-bold shadow-lg flex items-center gap-2">
                            <i v-if="loading" class="fa-solid fa-spinner fa-spin"></i>
                            <span v-else>実行</span>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>
    {% endraw %}

    <script>
        window.onload = () => {
            if (typeof Vue === 'undefined' || !Vue.createApp) {
                console.error("CRITICAL: Vue.js library failed to load even after window load. Check network.");
                alert("Vue.jsの読み込みに失敗しました。");
                return; 
            }
            
            const { createApp, ref, nextTick, onMounted } = Vue; 

            createApp({
                setup() {
                    const prompt = ref('');
                    const messages = ref([]);
                    const loading = ref(false);
                    const viewMode = ref('code'); 
                    const files = ref([]);
                    const currentFile = ref('');
                    const fileContent = ref('');
                    const previewUrl = ref('');
                    const chatLog = ref(null);
                    const codeBlock = ref(null);
                    const stats = ref({ currentPhase: 0, totalPhases: 0, l1: 0, l2: 0, l3: 0 });
                    const activeKit = ref('');

                    const scrollToBottom = () => nextTick(() => { if(chatLog.value) chatLog.value.scrollTop = chatLog.value.scrollHeight; });
                    const formatMessage = (content) => content ? content.replace(/\n/g, '<br>') : "";

                    const speak = (text) => {
                        if ('speechSynthesis' in window) {
                            const uttr = new SpeechSynthesisUtterance(text);
                            uttr.lang = 'ja-JP'; 
                            uttr.rate = 1.0;
                            window.speechSynthesis.speak(uttr);
                        }
                    };

                    const refreshFiles = async () => {
                        try {
                            const res = await fetch('/files');
                            const data = await res.json();
                            files.value = data.files || [];
                        } catch (e) { console.error(e); }
                    };

                    const loadFile = async (filename) => {
                        currentFile.value = filename;
                        try {
                            const res = await fetch(`/files/content?filename=${filename}`);
                            const data = await res.json();
                            fileContent.value = data.content || "";
                            viewMode.value = 'code';
                            
                            nextTick(() => {
                                if (codeBlock.value) {
                                    codeBlock.value.removeAttribute('data-highlighted');
                                    hljs.highlightElement(codeBlock.value); 
                                }
                            });

                            if (filename.endsWith('.html')) {
                                previewUrl.value = `/preview/${filename}`;
                            } else if (filename === 'app.py' || filename === 'main.py') {
                                previewUrl.value = `/preview/index.html`; 
                            }
                        } catch (e) { console.error(e); }
                    };

                    const generate = async () => {
                        if (!prompt.value || loading.value) return;
                        const userPrompt = prompt.value;
                        prompt.value = '';
                        messages.value.push({ role: 'user', content: userPrompt });
                        loading.value = true;
                        scrollToBottom();

                        try {
                            const res = await fetch('/generate', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify({ prompt: userPrompt })
                            });
                            
                            if (!res.ok) {
                                const errorText = await res.text();
                                throw new Error(`サーバーエラー (${res.status}): ${errorText.substring(0, 100)}...`);
                            }
                            
                            const data = await res.json();

                            if (data.stats) stats.value = data.stats;
                            if (data.kit_used) activeKit.value = data.kit_used;
                            if (data.logs) data.logs.forEach(log => messages.value.push({ role: 'system', content: log }));
                            
                            messages.value.push({ role: 'ai', content: data.message || "完了しました。" });

                            await refreshFiles();
                            if (data.main_file) loadFile(data.main_file);

                            speak("処理が完了しました。");

                        } catch (e) {
                            messages.value.push({ role: 'ai', content: `💥 エラー: ${e.message}` });
                        } finally {
                            loading.value = false;
                            scrollToBottom();
                        }
                    };

                    const reset = () => location.reload();

                    onMounted(() => {
                        refreshFiles();
                        messages.value.push({ role: 'system', content: 'Evo Studio 起動完了。' });
                    });

                    return { 
                        prompt, messages, loading, viewMode, files, currentFile, fileContent, previewUrl,
                        chatLog, codeBlock, stats, activeKit, generate, refreshFiles, loadFile, reset, formatMessage 
                    };
                }
            }).mount('#app');
        };
    </script>
</body>
</html>


