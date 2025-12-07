import logging
from ddgs import DDGS # パッケージ名変更に対応
from src.config import config

logger = logging.getLogger("SearchService")

class SearchService:
    """
    Web検索サービス (Cost Optimized)
    ページごとの要約(N回)をやめ、スニペット集約→最終回答(1回)に変更。
    """
    def __init__(self, client):
        self.client = client # Flash-Lite
        self.ddgs = DDGS()

    def research(self, query: str, max_results=3) -> str:
        logger.info(f"🔍 Searching for: '{query}'...")
        
        try:
            # 1. DuckDuckGoで検索 (無料)
            # bodyキーにスニペットが入っているのでこれを使う
            results = list(self.ddgs.text(query, max_results=max_results))
            if not results:
                return "No search results found."
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search Error: {str(e)}"

        # 2. コンテキストの集約 (ページにはアクセスしない)
        # 実際にページを開いてスクレイピングするのは時間がかかり、
        # スクレイピング対策で失敗することも多いため、検索エンジンの要約を信じる。
        
        context_data = ""
        for i, r in enumerate(results):
            title = r.get('title', 'No Title')
            link = r.get('href', '')
            snippet = r.get('body', '')
            context_data += f"Source {i+1}: {title}\nURL: {link}\nSummary: {snippet}\n\n"

        # 3. 1回だけLLMを呼び出してレポート作成
        prompt = f"""
        User Query: "{query}"

        Search Results:
        {context_data}

        Task: Summarize the search results to answer the user's query.
        Focus on technical details (libraries, code usage, installation).
        Output Format: Markdown
        """
        
        try:
            report = self.client.generate(prompt, "Role: Tech Researcher. Output: Concise technical summary.")
            return report
        except Exception as e:
            return f"Failed to generate report: {e}"