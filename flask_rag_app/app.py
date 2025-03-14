import os
import json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

# Import processing functions
from step1_step2_document_loading_chunking import process_uploaded_file
from step3_document_embedding import process_uploaded_embedding
from step4_vector_store import process_uploaded_vector_store
from step5_retrieval import query_retrieval

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx"}

def allowed_file(filename):
    """Check if the file type is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# === Route: Serve the Custom UI ===
@app.route("/")
def index():
    return render_template("index.html")

# === Route: Handle File Uploads ===
@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        print("DEBUG: No file part in request")
        return jsonify({"error": "No file part in request"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        print("DEBUG: No selected file")
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_ext = filename.rsplit(".", 1)[-1].lower()
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    print(f"DEBUG: Received file - Name: {filename}, Extension: {file_ext}")

    if file_ext not in ALLOWED_EXTENSIONS:
        print("DEBUG: Invalid file type detected:", file_ext)
        return jsonify({"error": f"Invalid file type: .{file_ext}"}), 400

    try:
        file.save(save_path)
        print(f"DEBUG: File {filename} saved successfully")

        # Process only the uploaded file
        process_uploaded_file(save_path)  # Step 1 & 2: Process & Chunk
        process_uploaded_embedding(save_path)  # Step 3: Generate Embeddings
        process_uploaded_vector_store(save_path)  # Step 4: Store in Vector DB

        return jsonify({"message": f"File '{filename}' uploaded and processed successfully!"})

    except Exception as e:
        print(f"ERROR: File upload failed - {str(e)}")
        return jsonify({"error": f"File upload failed: {str(e)}"}), 500

# === Route: Process Query and Get Response ===
@app.route("/query", methods=["POST"])
def query():
    try:
        data = request.json
        user_query = data.get("query", "").strip()  # Ensure it's a string

        if not user_query:
            return jsonify({"error": "Query must be a non-empty string."}), 400

        print(f"DEBUG: Received query: {user_query}")

        structured_response, generic_response = query_retrieval(user_query)

        return jsonify({
            "rag_output": structured_response,  # RAG pipeline output
            "llama_output": generic_response  # Same for now, can modify later
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
