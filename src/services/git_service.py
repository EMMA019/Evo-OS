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