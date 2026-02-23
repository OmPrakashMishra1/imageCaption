# 🖼️ ImageCaption — AI Image Captioning

An AI-powered image captioning system that generates human-readable descriptions for uploaded images using a **VGG16 + LSTM** deep learning architecture.

> **Mini Project** — Kalinga Institute of Industrial Technology (KIIT), Bhubaneswar

---

## ✨ Features

- 📷 Drag & drop image upload (JPG, JPEG, PNG)
- 🧠 Deep learning caption generation using VGG16 feature extraction + LSTM decoder
- ⚡ Flask REST API backend
- ⚛️ React frontend with a modern dark UI
- 🎨 Animated glassmorphism design with gradient accents

---

## 🛠️ Tech Stack

| Layer      | Technology             |
|------------|------------------------|
| Frontend   | React 18 (CDN), CSS3   |
| Backend    | Flask, Flask-CORS      |
| ML Model   | TensorFlow / Keras     |
| Feature Extraction | VGG16 (ImageNet) |
| Caption Decoder    | LSTM (Greedy decode) |
| Dataset    | Flickr8k               |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12** (required for TensorFlow compatibility)
- `pip` package manager

### Installation

```bash
# Clone the repo
git clone https://github.com/OmPrakashMishra1/imageCaption.git
cd imageCaption

# Install dependencies
pip install -r backend/requirements.txt
```

### Running the App

```bash
cd backend
py -3.12 app.py
```

Open **http://localhost:5000** in your browser — the React frontend is served directly by Flask.

---

## 📁 Project Structure

```
imageCaption/
├── backend/
│   ├── app.py              # Flask server + API routes + static file serving
│   ├── utils.py            # Model loading, VGG16 features, caption generation
│   ├── requirements.txt    # Python dependencies
│   └── model/
│       ├── best_model.h5   # Trained caption model weights
│       └── tokenizer.pkl   # Keras tokenizer (8485 vocab)
├── frontend/
│   ├── index.html          # React app (single-file, CDN-based)
│   └── style.css           # Dark UI with glassmorphism & animations
├── .gitignore
└── README.md
```

---

## 🔌 API

### `POST /caption`

Upload an image and receive a generated caption.

```bash
curl -X POST -F "image=@photo.jpg" http://localhost:5000/caption
```

**Response:**
```json
{ "caption": "a dog is running through the grass" }
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 👥 Team

| Name | Roll No. | Role |
|------|----------|------|
| Om Prakash Mishra | 23051283 | Tech Lead & Development Lead |
| Annada Shankar Maity | 23051245 | AI & ML |
| Kishalay Seren | 2305626 | AI & ML & Development |
| Suman Kumar Singha | 2305822 | R & D Lead |
| Priyanshu Dalei | 2305626 | R & D |
| Aryan Raj | 2305607 | R & D |

---

## 📝 Notes

- The model is trained on **Flickr8k**, a relatively small dataset. Caption accuracy improves significantly with larger datasets like Flickr30k or MS COCO.
- The `best_model.h5` file (~69 MB) is included in the repo. For very large models, consider using [Git LFS](https://git-lfs.github.com/).
