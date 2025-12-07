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