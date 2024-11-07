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
logger = logging.getLogger(__name__)

# Constants and initialization
TAMU_EVENTS_URL = "https://calendar.tamu.edu/live/json/events/group"
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://localhost:11434")

try:
    logger.info("Initializing application...")
    client = Client(host=MODEL_SERVER_URL)

    # Initialize Flask application
    app = Flask(__name__)
    CORS(app, origins=["https://frontend-f8j6.onrender.com", "http://localhost:5173"], 
         supports_credentials=False)

    # Initialize models
    logger.info("Loading models...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    llm_model = OllamaLLM(model="mistral", base_url=MODEL_SERVER_URL)

    # Load and preprocess data
    logger.info("Loading and preprocessing data...")
    events = load_data(TAMU_EVENTS_URL)
    preprocessed_events = preprocess_events(events)
    texts = [transform_event_to_sentence(e) for e in preprocessed_events]
    data_embeddings = embedding_model.encode(texts)
    logger.info("Initialization complete")

except Exception as e:
    logger.error(f"Initialization error: {str(e)}")
    preprocessed_events = None
    data_embeddings = None

@app.route("/")
def helloWorld():
    return "Hello, cross-origin-world!"

@app.route("/get_chatbot_response", methods=["GET"])
def get_chatbot_response():
    return query_one()

def query_one(retries=3):
    if preprocessed_events is None or data_embeddings is None:
        return jsonify({"error": "Server not properly initialized"}), 500
    
    try:
        query = request.args.get("query")
        if not query:
            return jsonify({"error": "No query provided"}), 400

        query_embedding = embedding_model.encode(query)
        similarities = cosine_similarity([query_embedding], data_embeddings).flatten()
        top_5_indices = similarities.argsort()[::-1][:5]
        top_5_events = [texts[i] for i in top_5_indices]
        context_text = "\n".join(f"Event {i+1}:\n{event}" for i, event in enumerate(top_5_events))
        
        prompt = f"""
        Answer the question based only on the following context, as briefly as possible:
        {context_text}
        ---
        Question: {query}
        Answer:
        """
        
        for attempt in range(retries):
            try:
                response = llm_model.invoke(prompt)
                logger.info(f"Query processed successfully")
                return jsonify({
                    "response": response,
                    "events": [preprocessed_events[i] for i in top_5_indices]
                })
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    raise
                
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 4999))
        logger.info(f"Starting server on port {port}")
        
        if os.getenv("ENVIRONMENT") == "production":
            # Production: use Waitress
            logger.info("Running in production mode with Waitress")
            serve(app, host="0.0.0.0", port=port)
        else:
            # Development: use Flask development server
            logger.info("Running in development mode")
            app.run(host="0.0.0.0", port=port, debug=True)
            
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise