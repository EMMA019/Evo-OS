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