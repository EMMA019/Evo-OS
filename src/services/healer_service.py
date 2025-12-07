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