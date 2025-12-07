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