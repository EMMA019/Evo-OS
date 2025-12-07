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