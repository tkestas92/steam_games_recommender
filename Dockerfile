FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt && \
    pip install --no-cache-dir gunicorn

COPY . .

RUN mkdir -p /app/models && cd /app/models && \
    curl -fL -o games.index "https://github.com/tkestas92/steam_games_recommender/releases/download/models/games.index" && \
    curl -fL -o app_ids.npy "https://github.com/tkestas92/steam_games_recommender/releases/download/models/app_ids.npy" && \
    curl -fL -o svd_item_factors.npy "https://github.com/tkestas92/steam_games_recommender/releases/download/models/svd_item_factors.npy" && \
    curl -fL -o game_ids.npy "https://github.com/tkestas92/steam_games_recommender/releases/download/models/game_ids.npy" && \
    curl -fL -o ncf_v28_final_metadata.pkl "https://github.com/tkestas92/steam_games_recommender/releases/download/models/ncf_v28_final_metadata.pkl" && \
    curl -fL -o ncf_v28_final.pth "https://github.com/tkestas92/steam_games_recommender/releases/download/models/ncf_v28_final.pth"

ENV MODEL_DIR=/app/models

EXPOSE 8080

CMD gunicorn wsgi:app --bind 0.0.0.0:${PORT:-8080} --timeout 180 --workers 1