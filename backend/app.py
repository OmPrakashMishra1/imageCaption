import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from utils import load_model_and_tokenizer, extract_features, generate_caption, generate_caption_beam

# ── App setup ────────────────────────────────────────────────────────────
BACKEND_DIR  = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BACKEND_DIR, "..", "frontend")

# Normalise so it works in Docker and locally
FRONTEND_DIR = os.path.normpath(FRONTEND_DIR)

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

UPLOAD_FOLDER = os.path.join(BACKEND_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model & tokenizer once at startup
caption_model, tokenizer = load_model_and_tokenizer()


# ── API Routes ───────────────────────────────────────────────────────────
@app.route("/caption", methods=["POST"])
def caption():
    """Accept an image file and return a generated caption."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        features = extract_features(filepath)
        method = request.args.get("method", "greedy").lower()
        if method == "beam":
            caption_text = generate_caption_beam(caption_model, tokenizer, features)
        else:
            caption_text = generate_caption(caption_model, tokenizer, features)
        return jsonify({"caption": caption_text, "method": method})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route("/health", methods=["GET"])
def health():
    """Simple health-check endpoint."""
    return jsonify({"status": "ok"})


# ── Serve React Frontend ─────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
