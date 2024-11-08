from typing import List, Optional
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
import uuid
import os
import sys
import requests
import pandas as pd
import numpy as np
import re
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import datasets
import time
import threading
from datetime import datetime, timedelta
import gc
import psutil
import h5py
from tqdm import tqdm
import atexit  # For registering cleanup functions
import xlsxwriter  # Required for Excel output (ensure this is installed)

# Additional FastAPI imports for shutdown handling
import uvicorn

# Load environment variables
load_dotenv()

API_KEY_ALLAM = os.getenv("IBM_WATSONX_API_KEY")
PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID")
API_URL_ALLAM = os.getenv("IBM_WATSONX_URL")
MODEL_ID_ALLAM = "sdaia/allam-1-13b-instruct"
API_KEY_CHATGPT = os.getenv("CHATGPT_API_KEY")
# Structure to hold quality scores and win counts

quality_data = {"allam_quality": [], "chatgpt_quality": [], "allam_wins": 0, "chatgpt_wins": 0}
# Function to save the quality data to an Excel file
def save_to_excel():
    data_df = pd.DataFrame({
        "Allam Quality": quality_data["allam_quality"],
        "ChatGPT Quality": quality_data["chatgpt_quality"]
    })
    
    # Add summary row for win counts
    summary_df = pd.DataFrame({
        "Allam Quality": ["Total Wins"],
        "ChatGPT Quality": [quality_data["allam_wins"] if quality_data["allam_wins"] > quality_data["chatgpt_wins"] else quality_data["chatgpt_wins"]]
    })
    
    data_df = pd.concat([data_df, summary_df], ignore_index=True)
    
    data_df.to_excel("game_quality_summary.xlsx", index=False)
    print("Quality data saved to game_quality_summary.xlsx")

# Register the save function to execute upon program termination
atexit.register(save_to_excel)

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class ModelManager:
    def __init__(self):
        self.allam_model = AllamModel(
            model_id=MODEL_ID_ALLAM,
            api_key=API_KEY_ALLAM,
            api_url=API_URL_ALLAM,
            project_id=PROJECT_ID
        )
        self.chatgpt_model = ChatGPTModel(api_key=API_KEY_CHATGPT)

    async def generate_response(self, model_name: str, conversation_history: List[str], last_letter: str, repetition_penalty: float) -> Optional[str]:
        if model_name == "Allam":
            return await self.allam_model.generate_response(conversation_history, last_letter, repetition_penalty)
        elif model_name == "ChatGPT":
            return await self.chatgpt_model.generate_response(conversation_history, last_letter, repetition_penalty)
        else:
            raise ValueError(f"Unknown model: {model_name}")

class AllamModel:
    def __init__(self, model_id: str, api_key: str, api_url: str, project_id: str):
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.project_id = project_id
        self.access_token = None
        self.token_expiry = None
        self.retry_count = 3
        self.retry_delay = 1  # seconds

    def get_access_token(self) -> str:
        """Retrieve access token using the API key."""
        current_time = datetime.now()
        
        # Check if token is still valid
        if self.access_token and self.token_expiry and current_time < self.token_expiry:
            return self.access_token

        token_url = "https://iam.cloud.ibm.com/identity/token"
        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        data = {
            "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
            "apikey": self.api_key
        }

        response = requests.post(token_url, headers=headers, data=data)
        if response.status_code == 200:
            token_data = response.json()
            self.access_token = token_data["access_token"]
            # Set token expiry to slightly less than actual expiry to ensure safety margin
            self.token_expiry = current_time + timedelta(minutes=55)  # Tokens typically valid for 1 hour
            return self.access_token
        else:
            raise Exception(f"Error retrieving token: {response.text}")

    async def generate_response(self, conversation_history: List[str], last_letter: str, repetition_penalty: float) -> Optional[str]:
        for attempt in range(self.retry_count):
            try:
                if not self.access_token:
                    self.get_access_token()

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}"
                }
                
                similar_verses = retrieve_similar_verses(" ".join(conversation_history[-2:]), last_letter, top_k=2)
                verses_text = "\n".join(similar_verses)
                prompt = f"""
                اكتب بيتاً شعرياً واحداً فقط يبدأ بحرف '{last_letter}' مناسباً للسياق التالي.
                يجب أن يكون البيت جديداً وغير مكرر.

                السياق الأخير:
                {verses_text}

                البيت يجب يكون بيت واحد فقط صدر وعجز أن يكون متناسقاً ويبدأ بحرف '{last_letter}' وغير مكرر.
                """

                data = {
                    "input": prompt,
                    "parameters": {
                        "decoding_method": "greedy",
                        "max_new_tokens": 200,
                        "min_new_tokens": 0,
                        "stop_sequences": [],
                        "repetition_penalty": repetition_penalty
                    },
                    "model_id": self.model_id,
                    "project_id": self.project_id
                }

                url = f"{self.api_url}/ml/v1/text/generation?version=2023-05-29"

                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 401:
                    self.access_token = None
                    self.get_access_token()
                    continue
                    
                if response.status_code == 200:
                    return response.json()['results'][0]['generated_text']
                else:
                    print(f"Error generating response from Allam model: {response.text}")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt == self.retry_count - 1:
                    print(f"Error in text generation after {self.retry_count} attempts: {str(e)}")
                    return None
                time.sleep(self.retry_delay * (attempt + 1))
                
            except Exception as e:
                print(f"Unexpected error in text generation: {str(e)}")
                return None

class ChatGPTModel:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.retry_count = 3
        self.retry_delay = 1  # seconds

    async def generate_response(self, conversation_history: List[str], last_letter: str, repetition_penalty: float) -> Optional[str]:
        for attempt in range(self.retry_count):
            try:
                similar_verses = retrieve_similar_verses(" ".join(conversation_history[-2:]), last_letter, top_k=2)
                verses_text = "\n".join(similar_verses)
                prompt = f"""
اكتب بيتاً شعرياً كاملاً يتكون من صدر وعجز، ويبدأ بحرف '{last_letter}' مناسباً للسياق التالي.
يجب أن يكون البيت جديداً وغير مكرر.

البيت يجب أن يكون كاملاً، ويتكون من صدر وعجز متناسقين، ويبدأ بحرف '{last_letter}' وغير مكرر.

يجب أن يكون البيت مكتوباً على النحو التالي:
البيت الشعري:
(صدر البيت)
(عجز البيت)
"""

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                        {"role": "system", "content": "أنت شاعر عربي متمرس في المساجلات الشعرية، وتكتب الأبيات الشعرية الكاملة التي تتكون من صدر وعجز."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.8,
                    "max_tokens": 300,
                    "top_p": 0.9
                }

                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
                if response.status_code == 200:
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"Error generating response from ChatGPT model: {response.text}")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt == self.retry_count - 1:
                    print(f"Error in text generation after {self.retry_count} attempts: {str(e)}")
                    return None
                time.sleep(self.retry_delay * (attempt + 1))

            except Exception as e:
                print(f"Unexpected error in text generation: {str(e)}")
                return None

# Update the GameSession class to store qualities in the game session
class GameSession(BaseModel):
    difficulty: str
    conversation_history: List[str] = []
    last_letter: Optional[str] = None
    required_verses: int = 6
    allam_verses_count: int = 0
    chatgpt_verses_count: int = 0
    is_game_over: bool = False
    game_result: Optional[str] = None
    created_at: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    allam_qualities: List[float] = []
    chatgpt_qualities: List[float] = []

class GameResponse(BaseModel):
    session_id: str
    message: str
    required_verses: int
    allam_verses: int
    chatgpt_verses: int
    last_letter: str
    allam_response: str
    chatgpt_response: str
    game_over: bool
    game_result: str
    allam_verse_quality: float
    chatgpt_verse_quality: float

class GameCreateRequest(BaseModel):
    difficulty: str

def normalize_letter(letter: str) -> Optional[str]:
    """Normalize Arabic letters."""
    if letter is None:
        return None
    letter = re.sub(r'[\u064B-\u0652]', '', letter)
    if letter in ['أ', 'إ', 'ء', 'ى', 'ئ']:
        return 'ا'
    return letter

def starts_with_letter(verse: str, letter: str) -> bool:
    """Check if verse starts with the given letter."""
    if letter is None or verse is None:
        return True
    verse = re.sub(r'[\u064B-\u0652]', '', verse)
    letter = re.sub(r'[\u064B-\u0652]', '', letter)
    verse_first = verse[0] if verse else None
    normalized_verse_first = normalize_letter(verse_first)
    normalized_letter = normalize_letter(letter)
    return normalized_verse_first == normalized_letter

def get_last_letter(verse: str) -> Optional[str]:
    """Get the last letter of a verse."""
    if not verse:
        return None
    verse = re.sub(r'[ًٌٍَُِّْـ\W]', '', verse)
    elongations = ['ا', 'و', 'ي', 'ها', 'ما', 'با', 'سا', 'دا', 'غا', 'فا', 'طا', 'جا', 'زا', 'شا', 'عا', 'قا', 'لا', 'نا', 'كا']
    endings_to_ignore = ['ة', 'ه', 'هم', 'هن', 'هما', 'وا']
    
    for elongation in elongations:
        if verse.endswith(elongation):
            return verse[-len(elongation)-1] if len(verse) > len(elongation) else verse[0]
    
    for ending in endings_to_ignore:
        if verse.endswith(ending):
            return verse[-len(ending)-1]
    
    return verse[-1]

def initialize_embeddings(model_name: str = "all-MiniLM-L6-v2", force_recreate: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Initialize embeddings with memory optimization."""
    h5_path = "data/embeddings.h5"
    verses_path = "data/verses.parquet"
    
    if os.path.exists(h5_path) and os.path.exists(verses_path) and not force_recreate:
        try:
            print("Loading existing embeddings...")
            with h5py.File(h5_path, 'r') as f:
                embeddings = f['embeddings'][:]
            ashaar_df = pd.read_parquet(verses_path)
            print(f"Loaded {len(embeddings)} existing embeddings successfully!")
            return embeddings, ashaar_df
        except Exception as e:
            print(f"Error loading existing embeddings: {str(e)}")
            print("Will create new embeddings...")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    os.makedirs(os.path.dirname(verses_path), exist_ok=True)
    
    print("Creating new embeddings...")
    dataset = datasets.load_dataset('arbml/ashaar')
    ashaar_df = pd.DataFrame(dataset['train'])
    
    def clean_verse(verse):
        if isinstance(verse, list):
            return " ".join(str(v) for v in verse)
        return str(verse)
    
    # Process data in smaller chunks
    chunk_size = 100
    ashaar_df['poem verses'] = ashaar_df['poem verses'].apply(clean_verse)
    
    with h5py.File(h5_path, 'w') as f:
        total_rows = len(ashaar_df)
        emb_func = SentenceTransformerEmbeddings(model_name=model_name)
        
        # Get embedding dimension from a sample
        sample_emb = emb_func.embed_documents([ashaar_df['poem verses'].iloc[0]])[0]
        embedding_dim = len(sample_emb)
        
        dset = f.create_dataset(
            'embeddings',
            shape=(0, embedding_dim),
            maxshape=(total_rows, embedding_dim),
            chunks=True,
            compression='gzip',
            compression_opts=9
        )
        
        current_idx = 0
        for i in tqdm(range(0, total_rows, chunk_size), desc="Creating embeddings"):
            gc.collect()
            
            chunk = ashaar_df['poem verses'].iloc[i:i + chunk_size].tolist()
            chunk_embeddings = emb_func.embed_documents(chunk)
            
            chunk_array = np.array(chunk_embeddings)
            dset.resize(current_idx + len(chunk_array), axis=0)
            dset[current_idx:current_idx + len(chunk_array)] = chunk_array
            
            current_idx += len(chunk_array)
            
            del chunk_embeddings
            del chunk_array
            gc.collect()
    
    ashaar_df.to_parquet(verses_path, index=False)
    
    with h5py.File(h5_path, 'r') as f:
        embeddings = f['embeddings'][:]
    
    return embeddings, ashaar_df

def retrieve_similar_verses(query: str, last_letter: Optional[str], top_k: int = 10) -> List[str]:
    """Retrieve similar verses from the dataset."""
    query_embedding = app.state.emb_func.embed_query(query)
    similarities = cosine_similarity([query_embedding], app.state.embeddings)[0]
   
    # Apply letter filter if specified
    if last_letter:
        mask = app.state.verses_df['poem verses'].apply(lambda x: starts_with_letter(x, last_letter))
        filtered_similarities = similarities * mask
    else:
        filtered_similarities = similarities
   
    top_indices = np.argsort(filtered_similarities)[-top_k*2:][::-1]
    selected_indices = np.random.choice(top_indices, min(top_k, len(top_indices)), replace=False)
   
    return app.state.verses_df.iloc[selected_indices]['poem verses'].tolist()

def calculate_verse_similarity(verse1: str, verse2: str) -> float:
    """Calculate the cosine similarity between two verses."""
    verse1_embedding = app.state.emb_func.embed_query(verse1)
    verse2_embedding = app.state.emb_func.embed_query(verse2)
    similarity = cosine_similarity([verse1_embedding], [verse2_embedding])[0][0]
    return similarity

game_sessions: dict[str, GameSession] = {}

def cleanup_old_sessions():
    """Remove expired game sessions."""
    current_time = datetime.now()
    timeout_threshold = timedelta(hours=1)
    inactivity_threshold = timedelta(minutes=30)
    
    sessions_to_remove = []
    for session_id, session in game_sessions.items():
        if session.is_game_over and (current_time - session.last_activity) > timeout_threshold:
            sessions_to_remove.append(session_id)
        elif (current_time - session.last_activity) > inactivity_threshold:
            sessions_to_remove.append(session_id)
            
    for session_id in sessions_to_remove:
        del game_sessions[session_id]

def update_session_activity(session_id: str):
    """Update last activity time for a session."""
    if session_id in game_sessions:
        game_sessions[session_id].last_activity = datetime.now()

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize app state and start cleanup timer."""
    # Initialize model manager
    app.state.model_manager = ModelManager()

    # Load embeddings
    print("Initializing embeddings...")
    app.state.embeddings, app.state.verses_df = initialize_embeddings()
    app.state.emb_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Start cleanup timer
    cleanup_timer = threading.Timer(900.0, cleanup_old_sessions)
    cleanup_timer.start()
    app.state.cleanup_timer = cleanup_timer

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    if hasattr(app.state, 'cleanup_timer'):
        app.state.cleanup_timer.cancel()
    
    # Clean up embeddings
    if hasattr(app.state, 'embeddings'):
        del app.state.embeddings
    if hasattr(app.state, 'verses_df'):
        del app.state.verses_df
    if hasattr(app.state, 'emb_func'):
        del app.state.emb_func
    
    gc.collect()

@app.post("/games", response_model=GameResponse)
async def create_game(request: GameCreateRequest):
    """Create a new game session."""
    if request.difficulty not in ["easy", "hard"]:
        raise HTTPException(status_code=400, detail="Invalid difficulty level. Choose 'easy' or 'hard'.")

    session_id = str(uuid.uuid4())
    required_verses = 6 if request.difficulty == "easy" else 10
    game_sessions[session_id] = GameSession(difficulty=request.difficulty, required_verses=required_verses)
    
    message = "لقد اخترت المستوى السهل." if request.difficulty == "easy" else "لقد اخترت المستوى الصعب."
    return GameResponse(
        session_id=session_id,
        message=message,
        required_verses=required_verses,
        allam_verses=0,
        chatgpt_verses=0,
        last_letter="",
        allam_response="",
        chatgpt_response="",
        game_over=False,
        game_result="",
        allam_verse_quality=0.0,
        chatgpt_verse_quality=0.0
    )

# Update the start_game endpoint to store qualities in session and global structure
@app.post("/games/{session_id}/start")
async def start_game(session_id: str):
    """Start a duel between Allam and ChatGPT, alternating turns."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")
    
    session = game_sessions[session_id]
    update_session_activity(session_id)
    
    if session.is_game_over:
        return GameResponse(
            session_id=session_id,
            message="اللعبة منتهية",
            required_verses=session.required_verses,
            allam_verses=session.allam_verses_count,
            chatgpt_verses=session.chatgpt_verses_count,
            last_letter=session.last_letter or "",
            allam_response="\n".join(session.conversation_history[::2]),
            chatgpt_response="\n".join(session.conversation_history[1::2]),
            game_over=True,
            game_result=session.game_result or "",
            allam_verse_quality=0.0,
            chatgpt_verse_quality=0.0
        )
    
    allam_verses = []
    chatgpt_verses = []
    
    while session.allam_verses_count + session.chatgpt_verses_count < session.required_verses:
        # Allam's turn
        allam_response = await app.state.model_manager.generate_response(
            "Allam",
            session.conversation_history,
            session.last_letter or "",
            1.0
        )
        if not allam_response:
            session.is_game_over = True
            session.game_result = "ChatGPT wins"
            return GameResponse(
                session_id=session_id,
                message="فشل Allam في توليد البيت",
                required_verses=session.required_verses,
                allam_verses=session.allam_verses_count,
                chatgpt_verses=session.chatgpt_verses_count,
                last_letter=session.last_letter or "",
                allam_response="\n".join(allam_verses),
                chatgpt_response="\n".join(chatgpt_verses),
                game_over=True,
                game_result=session.game_result,
                allam_verse_quality=0.0,
                chatgpt_verse_quality=0.0
            )
        session.conversation_history.append(allam_response)
        allam_verses.append(allam_response)
        session.last_letter = get_last_letter(allam_response)
        session.allam_verses_count += 1

        # Calculate verse quality for Allam
        allam_quality = calculate_verse_similarity(allam_response, session.conversation_history[-2] if len(session.conversation_history) > 1 else "")
        session.allam_qualities.append(allam_quality)
        quality_data["allam_quality"].append(allam_quality)

        if session.allam_verses_count + session.chatgpt_verses_count >= session.required_verses:
            break

        # ChatGPT's turn
        chatgpt_response = await app.state.model_manager.generate_response(
            "ChatGPT",
            session.conversation_history,
            session.last_letter or "",
            1.0
        )
        if not chatgpt_response:
            session.is_game_over = True
            session.game_result = "Allam wins"
            return GameResponse(
                session_id=session_id,
                message="فشل ChatGPT في توليد البيت",
                required_verses=session.required_verses,
                allam_verses=session.allam_verses_count,
                chatgpt_verses=session.chatgpt_verses_count,
                last_letter=session.last_letter or "",
                allam_response="\n".join(allam_verses),
                chatgpt_response="\n".join(chatgpt_verses),
                game_over=True,
                game_result=session.game_result,
                allam_verse_quality=0.0,
                chatgpt_verse_quality=0.0
            )
        session.conversation_history.append(chatgpt_response)
        chatgpt_verses.append(chatgpt_response)
        session.last_letter = get_last_letter(chatgpt_response)
        session.chatgpt_verses_count += 1

        # Calculate verse quality for ChatGPT
        chatgpt_quality = calculate_verse_similarity(chatgpt_response, session.conversation_history[-2])
        session.chatgpt_qualities.append(chatgpt_quality)
        quality_data["chatgpt_quality"].append(chatgpt_quality)
    
    # Determine winner and increment counts
    allam_avg_quality = sum(session.allam_qualities) / len(session.allam_qualities)
    chatgpt_avg_quality = sum(session.chatgpt_qualities) / len(session.chatgpt_qualities)
    if allam_avg_quality > chatgpt_avg_quality:
        quality_data["allam_wins"] += 1
    else:
        quality_data["chatgpt_wins"] += 1

    winner = "Allam" if allam_avg_quality > chatgpt_avg_quality else "ChatGPT"
    session.game_result = f"{winner} wins"
    session.is_game_over = True

    return GameResponse(
        session_id=session_id,
        message="اللعبة انتهت",
        required_verses=session.required_verses,
        allam_verses=session.allam_verses_count,
        chatgpt_verses=session.chatgpt_verses_count,
        last_letter=session.last_letter or "",
        allam_response="\n".join(allam_verses),
        chatgpt_response="\n".join(chatgpt_verses),
        game_over=True,
        game_result=session.game_result,
        allam_verse_quality=allam_avg_quality,
        chatgpt_verse_quality=chatgpt_avg_quality
    )
@app.get("/games/{session_id}", response_model=GameResponse)
async def get_game_status(session_id: str):
    """Get the current status of a game session."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")
        
    session = game_sessions[session_id]
    update_session_activity(session_id)
    
    return GameResponse(
        session_id=session_id,
        message="",
        required_verses=session.required_verses,
        allam_verses=session.allam_verses_count,
        chatgpt_verses=session.chatgpt_verses_count,
        last_letter=session.last_letter or "",
        allam_response="",
        chatgpt_response="",
        game_over=session.is_game_over,
        game_result=session.game_result or "",
        allam_verse_quality=0.0,
        chatgpt_verse_quality=0.0
    )

@app.delete("/games/{session_id}")
async def end_game(session_id: str):
    """End a game session."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")
        
    del game_sessions[session_id]
    return {"message": "تم إنهاء اللعبة بنجاح"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "نظام صحي", "message": "النظام يعمل بشكل طبيعي"}
@app.post("/generate_report")
async def generate_report():
    """Generate the Excel report manually and reset the quality data."""
    save_to_excel()  # Save current data to Excel
    # Reset quality data for new sessions
    quality_data["allam_quality"].clear()
    quality_data["chatgpt_quality"].clear()
    quality_data["allam_wins"] = 0
    quality_data["chatgpt_wins"] = 0
    return {"message": "Excel report generated and data reset successfully"}

if __name__ == "__main__":
    import uvicorn
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,
            limit_concurrency=10,
            limit_max_requests=1000,
            timeout_keep_alive=30,
            log_level="info"
        )
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)
