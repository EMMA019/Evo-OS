# ファイル名: Dockerfile
# 役割: Evo OS Core がコードを実行するための安全なサンドボックス環境定義
# ビルドコマンド: docker build -t evo-sandbox .

# 軽量かつ安定したPython環境をベースにする
FROM python:3.10-slim

# システムパッケージのインストール
# build-essential: C拡張のコンパイルに必要 (numpyなど)
# git, curl: 一般的なツール
# libxml2-dev, libxslt-dev: lxmlなどのパースライブラリ用
# nodejs, npm: React/Frontendのビルド用
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libxml2-dev \
    libxslt-dev \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# ワークスペースの設定（agent_core.pyのマウント先）
WORKDIR /workspace

# よく使われるPythonライブラリをプリインストール
# これにより、AIが生成したコードの ModuleNotFoundError を防ぎ、実行速度を上げる
# Qiskitなどの重いライブラリも含めることで「地獄級」タスクにも対応
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    flask \
    requests \
    beautifulsoup4 \
    lxml \
    matplotlib \
    pytest \
    scipy \
    scikit-learn \
    qiskit \
    fastapi \
    uvicorn \
    websockets

# コンテナが勝手に終了しないようにする（agent_core.pyが exec で入るため）
CMD ["sleep", "infinity"]