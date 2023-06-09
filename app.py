from pathlib import Path
from llama_index import GPTSimpleVectorIndex, Document, SimpleDirectoryReader,QuestionAnswerPrompt,LLMPredictor
import os
from langchain import OpenAI
import atexit
from flask import Flask, request
import json
import mimetypes
import pytesseract
from PIL import Image

app = Flask(__name__)

QA_PROMPT_TMPL = (
    "We have provided context information below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given this information, please answer the question as truthfully as possible using the provided text, and if the answer is not contained within the text below, say 'Not Found': {query_str}\n"
)
QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

ARTIFACTS_DIR="./artifacts"

def create_vector_index(dir_name):
    indexname = dir_name + "/index_file"
    document_exist=False
    documents=list()
    print("Parsing document")
    print(dir_name)
    #documents = SimpleDirectoryReader(dir_name, recursive=True).load_data()
    #llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003", max_tokens=1024))
    if "docs" in os.listdir(dir_name):
        document_exist=True
        documents = SimpleDirectoryReader(dir_name+"/docs/").load_data()
        print(documents)
        print(len(documents))
    if "images" in os.listdir(dir_name):
        document_exist=True
        documents=documents+ocr_with_chatgpt(dir_name+"/images/")
        print(len(documents))
    print("Creating an index")
    index = GPTSimpleVectorIndex.from_documents(documents)
    print("Persisting index")
    index.save_to_disk(indexname)
def ocr_with_chatgpt(dirName):
    images=os.listdir(dirName)
    documents=list()
    for image in images:
        output=pytesseract.image_to_string(Image.open(dirName+image))
        documents.append(Document(output))
    return documents

def run_query(question, dir_name):
    index_file = dir_name + "/index_file"
    print("Using index " + index_file)
    index = GPTSimpleVectorIndex.load_from_disk(index_file)
    response=index.query(question, text_qa_template=QA_PROMPT)
    return response.response

def cleanup_artifacts():
    os.system("rm -rf " + ARTIFACTS_DIR)

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.headers.get('sessionid')
    base_dir_name = ARTIFACTS_DIR + "/" + session_id
    print("Creating directory " + base_dir_name)
    try:
        os.mkdir(base_dir_name)
        os.mkdir(base_dir_name + "/images")
        os.mkdir(base_dir_name + "/docs")
    except FileExistsError:
        print("directory exists for session " + session_id)

    for uploaded_file, file_content in request.files.items(True):
        if file_content.content_type.startswith("image"):
            dir_name = base_dir_name + "/images"
        else:
            dir_name = base_dir_name + "/docs"

        file_name = dir_name + "/" + uploaded_file + mimetypes.guess_extension(file_content.content_type)
        file_content.content_type
        print("saving uploaded doc to " + file_name)
        file_content.save(file_name)

    create_vector_index(base_dir_name)
    return {
        "success": True
    }

@app.route("/query", methods=['POST'])
def get_query_response():
    request_body = json.loads(str(request.data, 'UTF-8'))
    print(request_body)
    question = request_body.get("question")
    session_id = request.headers.get('sessionId')

    if question is None:
        return {
            "answer": None,
            "error": "Please ask something"
        }

    return {
        "answer": run_query(question, ARTIFACTS_DIR + "/" + session_id),
        "error": None
    }

if __name__ == "__main__":
    atexit.register(cleanup_artifacts)
    try:
        os.mkdir(ARTIFACTS_DIR)
    except FileExistsError:
        print("artifacts directory already exists")
    app.run(host="0.0.0.0")
