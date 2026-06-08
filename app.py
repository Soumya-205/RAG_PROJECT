from flask import Flask, request, jsonify, render_template
from rag import ingest_document, query_rag
import os
from dotenv import load_dotenv

app=Flask(__name__)
load_dotenv()

app.config['UPLOAD_FOLDER']=os.getenv("UPLOAD_FOLDER")

ALLOWED_EXTENSIONS={"pdf"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload",methods=["POST"])
def upload():
    #checking if file is present or not
    if "file" not in request.files:
        return jsonify({"error":"No file provided"}),400
    file=request.files["file"]
    #Checking the file type
    if not allowed_file(file.filename):
        return jsonify({"error":"Only pdf files are allowed"}),400

    filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
    file.save(filepath)
    
    #handling errors
    try:
        ingest_document(filepath)
    except Exception as e:
        return jsonify({"error":f"Ingestion failed: {str(e)}"}),500
    
    return jsonify({"message":"Uploaded Successfully"})

@app.route("/query",methods=["POST"])
def query():
    data=request.get_json()
    question=data.get("question")
    if not question:
        return jsonify({"error":"Question cannot be empty"}),400
    
    #Handling errors
    try:
        answer=query_rag(question)
    except Exception as e:
        return jsonify({"error":f"Query failed: {str(e)}"}),500
    
    return jsonify({"answer": answer})

if __name__=="__main__":
    app.run(debug=True)

