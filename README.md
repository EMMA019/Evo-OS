<div align="center">

🧬 Evo OS

The Agent-First Development Orchestrator
<img width="1912" height="948" alt="image" src="https://github.com/user-attachments/assets/ffec4744-eb0e-4036-a2e1-6f172adbef3e" />


</div>

<a name="english"></a>

🧬 English

"Build software, don't just generate code."

Evo OS is a robust, cost-aware, and self-healing autonomous AI development framework designed to run locally with Docker sandboxing.

🧐 What is Evo OS?

Evo OS is an "Agent-First" development environment. Unlike simple coding assistants, Evo acts as a full engineering team. It plans the architecture, writes the code, verifies it using static analysis (AST), executes it in a sandbox to check for runtime errors, and fixes its own mistakes—all autonomously.

It addresses the critical flaws of current AI agents: Infinite Loops, Hidden Costs, and Environment Pollution.

✨ Key Features

🥇 BudgetGuard™ (Real-time Cost Control)

Evo tracks token usage in real-time and converts it to actual currency (JPY/USD). It strictly enforces budget limits per run (e.g., stops immediately if it exceeds 50 JPY), preventing unexpected API bills.

🛡️ Smart Healing & Loop Detection

Uses a multi-stage healing process (Patching -> Rewriting). Crucially, it includes logic to detect healing loops. If Evo gets stuck trying to fix the same error twice, it intelligently adapts its strategy or moves forward to prevent stalled processes.

🐳 Docker Sandboxing

All code execution happens inside a secure evo-sandbox Docker container. This ensures:

Security: No risk of malicious code running on your host (e.g., rm -rf / protection).

Consistency: Pre-installed heavy libraries (numpy, pandas, qiskit, scikit-learn, etc.) ensure fast and reproducible execution.

🧩 Kit System (Self-Expansion)

Evo selects specialized "Kits" (YAML-based domain knowledge) based on your prompt (e.g., "Make a Chrome Extension", "Create a Mahjong Game"). It can even generate new Kits for itself to learn new technologies on the fly.

🏗️ Architecture

The system operates on a microservices-like architecture orchestrated by the agent_core:

Planner (Architect Service): Breaks down user prompts into logical implementation phases.

Coding (Workspace Manager): Generates code and manages Git commits for every phase.

Verification (Verifier Service): Performs AST-based static analysis and security checks.

Healing (Healer Service): Autonomously fixes bugs using verify/runtime logs.

Runtime Test (Docker): Executes code in the sandbox to ensure it works.

Self-Improvement (Data Recorder): Logs successful tasks for future fine-tuning.

🚀 Quick Start

Prerequisites

Python 3.10+

Docker Desktop (Highly Recommended)

Google Gemini API Key

1. Installation

git clone [https://github.com/EMMA019/Evo-OS.git](https://github.com/EMMA019/Evo-OS.git)
cd Evo-OS
pip install -r requirements.txt


2. Configuration

Create a .env file in the root directory:

# .env
GEMINI_API_KEY="your_api_key_here"


3. Build Sandbox

Build the Docker runtime environment (First time only).
Includes heavy libraries like Scikit-learn & Qiskit for robust execution.

docker build -t evo-sandbox .


4. Launch

python server.py


Open http://localhost:5000 in your browser to access the Retro IDE.

🎮 Usage

Enter your request in the chat bar.

e.g., "Create a Python script to visualize Bitcoin price trends using Streamlit."

e.g., "Make a Chrome Extension that summarizes web pages."

Click Execute.

Watch Evo plan, code, test, and fix in real-time logs.

Browse the generated files in the left panel or click Download to get the artifacts.

<a name="japanese"></a>

🇯🇵 Japanese

"AIエージェントの無限ループと青天井のコストにさようなら。"

Evo OSは、計画・実装・検証・自己修復を自律的に行う、堅牢で経済的な次世代AI開発フレームワークです。

🧐 Evo OS とは？

単にコードを生成するだけではありません。Evoは以下のパイプラインを自律的に回し、動くソフトウェアを完成させます。

Architect: ユーザーの意図を理解し、実装フェーズを策定。

Coding: 適切なファイル構造でコードを生成。

Verification: ASTを用いた静的解析とセキュリティチェック。

Healing: エラーを検知し、自律的に修正パッチを適用（無限ループ検知機能付き）。

Runtime Test: Dockerサンドボックス内で実際にコードを実行し、動作を保証。

✨ なぜ Evo OS なのか？ (Key Features)

🥇 1. 鉄壁のコスト管理 (BudgetGuard™)

LLMのAPIコストをリアルタイムに日本円で計算し、設定予算（例: 50円）を超えた瞬間に停止します。「朝起きたらAPI利用料で破産していた」というAI開発のリスクをゼロにします。

🛡️ 無限ループ防止 (Healer Logic)

AIエージェント最大の弱点である「修正の無限ループ」を検知するロジックを標準搭載。同じエラーで詰まった場合、戦略的にループを脱出またはスキップし、タスク全体の完了を優先します。

🐳 安全な実行環境 (Docker Sandbox)

生成されたコードは、ホストマシンではなく隔離されたDockerコンテナ（evo-sandbox）内で実行されます。rm -rf / のような危険なコードや、環境を汚染するライブラリインストールからあなたのPCを守ります。Qiskitなどの重いライブラリもプリインストール済みで、即座に実行可能です。

🧠 自己拡張システム (Kit System)

「Chrome拡張機能を作って」「麻雀ゲームを作って」といった要望に対し、最適な専門知識（Kit）を自動でロードします。さらに、AI自身が新しいKitを生成し、知識を拡張していくことが可能です。

🚀 クイックスタート

前提条件

Python 3.10+

Docker Desktop (推奨)

Google Gemini API Key

1. インストール

git clone [https://github.com/EMMA019/Evo-OS.git](https://github.com/EMMA019/Evo-OS.git)
cd Evo-OS
pip install -r requirements.txt


2. 環境設定

.env ファイルを作成し、APIキーを設定します。

# .env
GEMINI_API_KEY="your_api_key_here"


3. サンドボックスの構築

実行環境となるDockerイメージをビルドします（初回のみ）。
※ データサイエンス系ライブラリもプリインストールされるため、少し時間がかかります。

docker build -t evo-sandbox .


4. 起動

python server.py


ブラウザで http://localhost:5000 にアクセスしてください。「Retro IDE」が表示されます。

🎮 使い方

画面下部のチャットボックスに作りたいものを入力します。

例: 「Streamlitで株価可視化アプリを作って」

例: 「Chrome拡張機能で、閲覧中のページの要約をするツールを作って」

実行をクリック。

Evoが思考を開始し、ファイル生成、コード記述、テスト、修正を行う様子がリアルタイムでログに表示されます。

完了後、Download ボタンで生成物を取得できます。

🤝 Contributing / 貢献について

Evo OS is an experimental project exploring the potential of autonomous agents. Issues and Pull Requests are always welcome!
Evo OSは、自律型エージェントの可能性を探求する実験的プロジェクトです。IssueやPull Requestはいつでも歓迎します！

📄 License

MIT License
