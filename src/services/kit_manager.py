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