FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -c "from transformers import TFBertModel, BertTokenizer; \
    TFBertModel.from_pretrained('bert-base-uncased'); \
    BertTokenizer.from_pretrained('bert-base-uncased')"
RUN python -c "from tensorflow.keras.applications import ResNet50; \
    ResNet50(weights='imagenet', include_top=False)"
# RUN python -c "from sentence_transformers import CrossEncoder; \
    # CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]