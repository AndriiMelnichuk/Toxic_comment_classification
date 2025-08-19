# Toxic Comment Classification

A robust machine learning pipeline for detecting toxic comments using FastText embeddings and PyTorch. This project features advanced preprocessing, data augmentation via back-translation, and a REST API for real-time predictions.

---

## 🚀 Features

- **FastText Embeddings:** Leverages pre-trained word vectors for rich text representation.
- **Custom PyTorch Model:** Multi-label classification for various toxicity types.
- **Advanced Preprocessing:** Regex cleaning, lemmatization (spaCy), stopword removal, and more.
- **Back-Translation Augmentation:** Uses Google Translate for generating paraphrased samples.
- **Batch & Single Prediction API:** FastAPI endpoints for scalable inference.
- **Docker Support:** Easy deployment with Docker.
- **TensorBoard Logging:** Track training metrics and visualize model performance.

---

## 🗂️ Project Structure

```
app/                # Source code (API, model, preprocessing, etc.)
data/
  raw/              # Raw datasets
  processed/        # Preprocessed datasets
models/             # FastText embeddings & trained model weights
runs/               # TensorBoard logs
scripts/            # Bash scripts for setup and running
Dockerfile          # Containerization
requirements.txt    # Python dependencies
README.md           # Project documentation
```

---

## ⚡ Quickstart

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Download datasets

```sh
bash scripts/data_load.sh
```

### 3. Download FastText embeddings

```sh
bash scripts/fasttext_load.sh
```

### 4. Run the API server

```sh
bash scripts/app_up.sh
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## 🧠 Model Overview

- **Input:** English comment text
- **Output:** Multi-label prediction for:
  - toxic
  - severe_toxic
  - obscene
  - threat
  - insult
  - identity_hate

Model weights: [`models/toxic_model_weights.pth`](models/toxic_model_weights.pth)  
Embeddings: [`models/cc.en.300.bin`](models/cc.en.300.bin)

---

## 🛠️ API Endpoints

- `POST /predict`  
  Predict toxicity for a single comment.
  ```json
  { "text": "your comment here" }
  ```

- `POST /predict-batch`  
  Predict toxicity for multiple comments.
  ```json
  { "texts": ["comment one", "comment two"] }
  ```

- `GET /health`  
  Health check endpoint.

See implementation in [`app/main.py`](app/main.py).

---

## 🧹 Preprocessing & Augmentation

- **Preprocessing:**  
  See [`app/preprocessing.py`](app/preprocessing.py) for regex cleaning, punctuation handling, lemmatization, and stopword removal.

- **Back-Translation:**  
  Augments training data by translating comments to other languages and back to English.

---

## 📦 Docker

Build and run the project in a container:

```sh
docker build -t toxic-comment-classifier .
docker run -p 8000:8000 toxic-comment-classifier
```

---

## 📊 Training & Evaluation

- Training scripts and configuration in [`app/modeling.py`](app/modeling.py)
- TensorBoard logs in [`runs/`](runs/)

---

## 👤 Author

- Leglone

---

## 📄 License

MIT

---

## 💡 References

- [FastText](https://fasttext.cc/)
- [spaCy](https://spacy.io/)
- [Polars](https://pola.rs/)
