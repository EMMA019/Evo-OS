import logging
from typing import Dict, Optional

logger = logging.getLogger("KitGenService")

class KitGenService:
    """
    ユーザーの要望に基づいて、Evo自身の拡張プラグイン(Kit YAML)を生成するサービス。
    自己進化の中核を担う。
    """
    def __init__(self, client):
        self.client = client # Smart Client (Pro/Flash) を使用

    def generate_kit(self, user_prompt: str) -> str:
        """
        ユーザーの自然言語記述から、有効なKit YAMLを生成する
        """
        system_prompt = """
        あなたはAIエージェント「Evo」の機能拡張エンジニアです。
        ユーザーの要望に基づき、Evoが特定のタスクを遂行するための「Kit（専門知識定義ファイル）」をYAML形式で作成してください。

        【Kitの構成要素】
        1. id: 一意の識別子 (英数字とアンダースコア)
        2. name: わかりやすい名前
        3. description: 何をするKitか
        4. triggers: このKitが発動すべきキーワードとサンプルプロンプト
        5. blueprint: 推奨技術スタック、主要コンポーネント、ファイル構成
        6. resources: **最重要**。AIに与えるドメイン知識、設計思想、ベストプラクティス。

        【出力ルール】
        - 必ず有効なYAML形式のみを出力すること。
        - マークダウンのコードブロック (```yaml ... ```) で囲むこと。
        - `domain_knowledge` は具体的かつ専門的に書くこと（ライブラリの正しい使い方、落とし穴、設計パターンなど）。

        【出力例】
        ```yaml
        id: "discord_bot_py"
        name: "Discord Bot Builder"
        description: "discord.pyを使用した高機能Bot開発キット"
        triggers:
          keywords: ["discord", "bot", "ディスコード"]
          sample_prompts: ["サーバー管理Botを作って"]
        blueprint:
          suggested_tech_stack: ["Python 3.10", "discord.py", "python-dotenv"]
          core_components: ["Event Listener", "Command Tree", "Cog System"]
          expected_file_structure:
            - "main.py"
            - "cogs/general.py"
            - ".env"
        resources:
          domain_knowledge: |
            discord.py 2.0以降では `Intents` の設定が必須です。
            大規模なBotの場合は `Cogs` 機能を使ってコマンドを分割管理してください。
            トークンは必ず環境変数から読み込むこと。
        ```
        """

        prompt = f"以下の要望を満たすKitを作成してください:\n{user_prompt}"
        
        logger.info("🧠 Generating new Kit definition...")
        response = self.client.generate(prompt, system_prompt)
        
        # クリーニング (Markdown除去)
        yaml_content = response.replace("```yaml", "").replace("```", "").strip()
        return yaml_content