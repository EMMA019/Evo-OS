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