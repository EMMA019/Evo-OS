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