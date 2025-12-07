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