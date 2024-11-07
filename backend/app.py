import json
import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_ollama import OllamaLLM
from waitress import serve
from ollama import Client
from preprocess import load_data, preprocess_events, transform_event_to_sentence

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

TAMU_EVENTS_URL = "https://calendar.tamu.edu/live/json/events/group"
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:11434")
client = Client(host=MODEL_SERVER_URL)

# initialize flask application
app = Flask(__name__)
CORS(app, origins=["https://frontend-f8j6.onrender.com", "http://localhost:5173"], supports_credentials=False)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
llm_model = OllamaLLM(model="mistral", base_url=MODEL_SERVER_URL)
preprocessed_events = None
data_embeddings = None
texts = []  # Define texts at module level

@app.route("/")
def helloWorld():
    return "Hello, cross-origin-world!"

@app.route("/get_chatbot_response", methods=["GET"])
def get_chatbot_response():
    return query_one()

def query_one(retries=3):
    if preprocessed_events is not None and data_embeddings is not None:
        # get query and create embedding
        query = request.args.get("query")
        query_embedding = embedding_model.encode(query)
        # compute cosine similarity between query and all data embeddings
        similarities = cosine_similarity([query_embedding], data_embeddings).flatten()
        top_5_indices = similarities.argsort()[::-1][:5]
        top_5_events = [texts[i] for i in top_5_indices]
        context_text = "\n".join(f"Event {i+1}:\n{event}" for i, event in enumerate(top_5_events))
        logging.info(f"Context: {context_text}")
        
        # prompt engineering
        prompt = f"""
        Answer the question based only on the following context, as briefly as possible:
        {context_text}
        ---
        Question: {query}
        Answer:
        """
        response = llm_model.invoke(prompt)
        logging.info(f"Response: {response}")
        return jsonify({
            "response": response,
            "events": [preprocessed_events[i] for i in top_5_indices]
        })
    else:
        return jsonify({"error": "Fetch errors."}), 500

if __name__ == "__main__":
    try:
        # Initialize models and data
        logging.info("Initializing models...")
        events = load_data(TAMU_EVENTS_URL)
        preprocessed_events = preprocess_events(events)
        for e in preprocessed_events:
            sentence = transform_event_to_sentence(e)
            texts.append(sentence)
        data_embeddings = embedding_model.encode(texts)
        logging.info("Models initialized successfully")

        # Get port from environment variable, default to 4999
        port = int(os.getenv("PORT", 4999))
        logging.info(f"PORT environment variable: {os.getenv('PORT')}")

        if os.getenv("FLASK_ENV") == "production":
            # Production: use Waitress
            logging.info(f"Starting production server on port {port}")
            serve(app, host="0.0.0.0", port=port)
        else:
            # Development: use Flask's built-in server
            logging.info(f"Starting development server on port {port}")
            app.run(host="0.0.0.0", port=port, debug=True)

    except Exception as e:
        logging.error(f"Failed to start server: {e}")