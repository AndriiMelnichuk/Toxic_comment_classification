FROM python:3.12-slim

WORKDIR /toxic_comment_classification

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    gzip \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY app/ app/
COPY models/ models/
COPY scripts/ scripts/

RUN pip install --no-cache-dir -r requirements.txt
RUN bash scripts/fasttext_load.sh
RUN python -m nltk.downloader stopwords
RUN python -m spacy download en_core_web_sm

EXPOSE 8000

CMD [ "bash", "scripts/app_up.sh" ]