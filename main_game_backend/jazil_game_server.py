from typing import List, Optional, Dict, Tuple
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
from langchain.embeddings.base import Embeddings
from sklearn.metrics.pairwise import cosine_similarity
import datasets
import time
from threading import Timer
from datetime import datetime, timedelta
import gc
import psutil
import h5py
from tqdm import tqdm
import statistics
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import xlsxwriter
from datetime import datetime
import os

# Load environment variables
load_dotenv()

API_KEY = os.getenv("IBM_WATSONX_API_KEY")
PROJECT_ID = os.getenv("IBM_WATSONX_PROJECT_ID")
API_URL = os.getenv("IBM_WATSONX_URL")
MODEL_ID = "sdaia/allam-1-13b-instruct"

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class ModelManager:
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

    def generate_text(self, prompt: str,repetition_penalty:float) -> Optional[str]:
        """Generate text using IBM WatsonX API with token count logging."""
        estimated_tokens = estimate_token_count(prompt)
        if estimated_tokens > 3500:  # Leave some room for generation
            print(f"Warning: Estimated token count ({estimated_tokens}) is close to limit")
        for attempt in range(self.retry_count):
            try:
                if not self.access_token:
                    self.get_access_token()

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_token}"
                }
                
                # Corrected request payload structure
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

                # Add version parameter to URL
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

class PerformanceMetrics(BaseModel):
    response_time: float
    verse_similarity: float
    context_similarity: float
# Add these new models
class GameAnalytics(BaseModel):
    session_id: str
    verses: List[str]
    metrics: List[PerformanceMetrics]
    timestamp: datetime
    game_result: str
    difficulty: str

class AnalyticsReport(BaseModel):
    excel_path: str
    graphs_paths: List[str]
    summary: dict
class GameResponse(BaseModel):
    session_id: str
    message: str
    required_verses: int
    current_verses: int
    strikes: int
    last_letter: str
    ai_response: str
    game_over: bool
    game_result: str
    performance_metrics: Optional[PerformanceMetrics] = None

class GameSession(BaseModel):
    difficulty: str
    conversation_history: List[str] = []
    last_letter: Optional[str] = None
    required_verses: int = 6
    user_verses_count: int = 0
    strikes: int = 0
    is_game_over: bool = False
    game_result: Optional[str] = None
    created_at: datetime = datetime.now()
    last_activity: datetime = datetime.now()
    performance_history: List[PerformanceMetrics] = []
    analytics: Optional[GameAnalytics] = None
class GameCreateRequest(BaseModel):
    difficulty: str

class VerseRequest(BaseModel):
    verse: str
class GameAnalytics(BaseModel):
    session_id: str
    verses: List[str]
    metrics: List[PerformanceMetrics]
    timestamp: datetime
    game_result: str
    difficulty: str

class AnalyticsReport(BaseModel):
    excel_path: str
    graphs_paths: List[str]
    summary: dict
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

def verify_verse(verse: str, last_letter: Optional[str] = None) -> tuple[bool, str]:
    """Verify if the verse is valid Arabic poetry."""
    print(f"Verifying verse: '{verse}'")  # Debug print
    
    # Basic validation
    if not verse:
        print("Verse is empty")
        return False, "البيت فارغ"
        
    cleaned_verse = verse.strip()
    if len(cleaned_verse) < 10:
        print(f"Verse too short: {len(cleaned_verse)} characters")
        return False, "البيت قصير جداً (يجب أن يكون أكثر من 10 أحرف)"
        
    if not re.search(r'[\u0600-\u06FF]', cleaned_verse):
        print("No Arabic characters found")
        return False, "يجب أن يحتوي البيت على حروف عربية"

    try:
        # Simplified prompt for better response
        prompt = f"""
                 قيم البيت التالي وأجب بـ "صحيح" أو "غير صحيح" فقط وفق الشروط الآتية:
                     أنت خبير متخصص في الشعر العربي الكلاسيكي. مهمتك هي تحليل النص المقدم وتحديد ما إذا كان بيتًا شعريًا صحيحًا أم لا.
            قم بتقييم النص التالي بدقة شديدة وفقًا للمعايير الآتية:
            1. الشكل: هل يتكون النص من صدر وعجز متوازنين؟
            2. الوزن الشعري: هل يتبع النص أحد البحور الشعرية العربية المعروفة؟
            3. القافية: هل توجد قافية واضحة ومناسبة في نهاية البيت؟
            4. اللغة: هل يستخدم النص لغة عربية فصحى وتراكيب شعرية تقليدية؟
            5. المعنى: هل للنص معنى شعري واضح ومتماسك؟
            6. الأصالة: هل يبدو النص كأنه من الشعر العربي الكلاسيكي وليس مجرد جملة عادية أو نصًا حديثًا؟
            7. هل البيت المذكور يحتوي فقط حرف؟             
            8. هل البيت المذكور يحتوي على قافية ومعنى شعري ؟
            النص المراد تقييمه:

        البيت:
        {cleaned_verse}
"""
        print(f"this is the cleaned verse: {cleaned_verse}")
        verification_response = app.state.model_manager.generate_text(prompt,1.0)
        print(f"Model response: {verification_response}")  # Debug print
        
        if not verification_response:
            print("No response from model")
            return False, "فشل التحقق من صحة البيت"
            
        # Fixed logic: Only return True if exactly "صحيح" is in the response
        is_valid = "صحيح" in verification_response and "غير صحيح" not in verification_response
        
        if is_valid:
            print("Verse verified as correct")
            return True, ""
        else:
            print("Verse verified as incorrect")
            return False, "البيت غير صحيح. يجب أن يكون بيت شعر عربي فصيح"
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return False, "حدث خطأ أثناء التحقق من البيت"
def calculate_verse_metrics(verse: str, context: List[str], embeddings_func) -> Tuple[float, float]:
    """Calculate similarity metrics for a verse."""
    # Get embeddings
    verse_embedding = embeddings_func.embed_query(verse)
    
    # Calculate similarity with traditional verses
    traditional_verses = app.state.verses_df['poem verses'].tolist()[:1000]  # Sample for efficiency
    traditional_embeddings = app.state.embeddings[:1000]
    verse_similarities = cosine_similarity([verse_embedding], traditional_embeddings)[0]
    verse_similarity = np.mean(verse_similarities)
    
    # Calculate similarity with conversation context
    if context:
        context_embeddings = embeddings_func.embed_documents(context)
        context_similarities = cosine_similarity([verse_embedding], context_embeddings)[0]
        context_similarity = np.mean(context_similarities)
    else:
        context_similarity = 0.0
        
    return verse_similarity, context_similarity
def generate_verse(conversation_history: List[str], last_letter: str) -> Tuple[str, PerformanceMetrics]:
    """Generate a verse with performance metrics."""
    start_time = time.time()
    
    try:
        recent_context = conversation_history[-2:] if len(conversation_history) > 2 else conversation_history
        context_text = " ".join(recent_context)
        similar_verses = retrieve_similar_verses(context_text, last_letter, top_k=2)
        verses_text = "\n".join(similar_verses)
        
        prompt = f"""
        اكتب بيتاً شعرياً واحداً فقط يبدأ بحرف '{last_letter}' مناسباً للسياق التالي.
        يجب أن يكون البيت جديداً وغير مكرر.

        السياق الأخير:
        {verses_text}

        البيت يجب يكون بيت واحد فقط صدر وعجز أن يكون متناسقاً ويبدأ بحرف '{last_letter}' وغير مكرر.
        """
        
        generated_verse = app.state.model_manager.generate_text(prompt, 1.0)
        
        # Calculate metrics
        verse_similarity, context_similarity = calculate_verse_metrics(
            generated_verse, 
            conversation_history,
            app.state.emb_func
        )
        
        response_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            response_time=response_time,
            verse_similarity=verse_similarity,
            context_similarity=context_similarity
        )
        
        return generated_verse, metrics
        
    except Exception as e:
        print(f"Error in verse generation: {str(e)}")
        return generate_verse(conversation_history, last_letter)

# Helper function to estimate token count (rough approximation)
def estimate_token_count(text: str) -> int:
    """Rough estimation of token count for Arabic text."""
    # Split on whitespace and punctuation
    words = re.findall(r'\b\w+\b', text)
    # Rough estimate: each Arabic word is about 1.5 tokens on average
    return int(len(words) * 1.5)

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
    app.state.model_manager = ModelManager(
        model_id=MODEL_ID,
        api_key=API_KEY,
        api_url=API_URL,
        project_id=PROJECT_ID
    )

    # Load embeddings
    print("Initializing embeddings...")
    app.state.embeddings, app.state.verses_df = initialize_embeddings()
    app.state.emb_func = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Start cleanup timer
    cleanup_timer = Timer(900.0, cleanup_old_sessions)
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
        current_verses=0,
        strikes=0,
        last_letter="",
        ai_response="",
        game_over=False,
        game_result=""
    )

@app.post("/games/{session_id}/verses", response_model=GameResponse)
async def submit_verse(session_id: str, verse: VerseRequest):
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")

    session = game_sessions[session_id]
    update_session_activity(session_id)
    
    if session.is_game_over:
        return GameResponse(
            session_id=session_id,
            message="اللعبة منتهية",
            required_verses=session.required_verses,
            current_verses=session.user_verses_count,
            strikes=session.strikes,
            last_letter=session.last_letter or "",
            ai_response="",
            game_over=True,
            game_result=session.game_result or ""
        )
        
    verse_text = verse.verse.strip()
    is_first_verse = len(session.conversation_history) == 0

    if verse_text in session.conversation_history:
        session.strikes += 1
        return GameResponse(
            session_id=session_id,
            message="هذا البيت مكرر! يجب عليك استخدام بيت جديد",
            required_verses=session.required_verses,
            current_verses=session.user_verses_count,
            strikes=session.strikes,
            last_letter=session.last_letter or "",
            ai_response="",
            game_over=session.strikes >= 3,
            game_result="lose" if session.strikes >= 3 else ""
        )
    
    if not is_first_verse and session.last_letter:
        if not starts_with_letter(verse_text, session.last_letter):
            session.strikes += 1
            return GameResponse(
                session_id=session_id,
                message=f"يجب أن يبدأ البيت بحرف {session.last_letter}",
                required_verses=session.required_verses,
                current_verses=session.user_verses_count,
                strikes=session.strikes,
                last_letter=session.last_letter or "",
                ai_response="",
                game_over=session.strikes >= 3,
                game_result="lose" if session.strikes >= 3 else ""
            )
    
    is_valid_verse, error_message = verify_verse(verse_text, session.last_letter)
    if not is_valid_verse:
        session.strikes += 1
        return GameResponse(
            session_id=session_id,
            message=error_message,
            required_verses=session.required_verses,
            current_verses=session.user_verses_count,
            strikes=session.strikes,
            last_letter=session.last_letter or "",
            ai_response="",
            game_over=session.strikes >= 3,
            game_result="lose" if session.strikes >= 3 else ""
        )

    session.conversation_history.append(verse_text)
    user_last_letter = get_last_letter(verse_text)

    ai_response, metrics = generate_verse(session.conversation_history, user_last_letter)
    session.conversation_history.append(ai_response)
    session.last_letter = get_last_letter(ai_response)
    session.user_verses_count += 1
    
    session.performance_history.append(metrics)
    
    avg_response_time = statistics.mean([m.response_time for m in session.performance_history])
    avg_verse_similarity = statistics.mean([m.verse_similarity for m in session.performance_history])
    avg_context_similarity = statistics.mean([m.context_similarity for m in session.performance_history])
    
    print(f"""
    Performance Metrics:
    - Response Time: {metrics.response_time:.2f}s (avg: {avg_response_time:.2f}s)
    - Verse Similarity: {metrics.verse_similarity:.2%} (avg: {avg_verse_similarity:.2%})
    - Context Similarity: {metrics.context_similarity:.2%} (avg: {avg_context_similarity:.2%})
    """)
    
    if session.user_verses_count >= session.required_verses:
        session.is_game_over = True
        session.game_result = "win"
        message = f"مبروك! أكملت {session.user_verses_count} أبيات بنجاح"
    else:
        message = f"بيت شعري صحيح! عدد الأبيات: {session.user_verses_count}/{session.required_verses}"

    return GameResponse(
        session_id=session_id,
        message=message,
        required_verses=session.required_verses,
        current_verses=session.user_verses_count,
        strikes=session.strikes,
        last_letter=session.last_letter or "",
        ai_response=ai_response,
        game_over=session.is_game_over,
        game_result=session.game_result or "",
        performance_metrics=metrics
    )

@app.get("/games/{session_id}/metrics")
async def get_game_metrics(session_id: str):
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")
        
    session = game_sessions[session_id]
    
    if not session.performance_history:
        return {"message": "No metrics available for this session"}
    
    metrics = {
        "response_times": {
            "average": statistics.mean([m.response_time for m in session.performance_history]),
            "min": min([m.response_time for m in session.performance_history]),
            "max": max([m.response_time for m in session.performance_history])
        },
        "verse_similarities": {
            "average": statistics.mean([m.verse_similarity for m in session.performance_history]),
            "min": min([m.verse_similarity for m in session.performance_history]),
            "max": max([m.verse_similarity for m in session.performance_history])
        },
        "context_similarities": {
            "average": statistics.mean([m.context_similarity for m in session.performance_history]),
            "min": min([m.context_similarity for m in session.performance_history]),
            "max": max([m.context_similarity for m in session.performance_history])
        },
        "total_verses": len(session.performance_history)
    }
    
    return metrics
def calculate_performance_rating(metrics_df: pd.DataFrame) -> str:
    """Calculate overall performance rating based on metrics."""
    avg_verse_sim = metrics_df['verse_similarity'].mean()
    avg_context_sim = metrics_df['context_similarity'].mean()
    avg_response = metrics_df['response_time'].mean()
    
    # Weight each factor
    verse_weight = 0.4
    context_weight = 0.3
    response_weight = 0.3
    
    # Normalize response time (lower is better)
    response_score = 1 - (avg_response / 10)  # Assuming 10 seconds is maximum acceptable
    response_score = max(0, min(1, response_score))
    
    # Calculate weighted score
    total_score = (
        avg_verse_sim * verse_weight +
        avg_context_sim * context_weight +
        response_score * response_weight
    )
    
    # Convert to rating
    if total_score >= 0.8:
        return "Excellent"
    elif total_score >= 0.6:
        return "Good"
    elif total_score >= 0.4:
        return "Average"
    else:
        return "Needs Improvement"
@app.get("/analytics/report/{session_id}")
async def generate_session_report(session_id: str):
    """Generate comprehensive analytics report for a game session."""
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Game session not found.")
    
    session = game_sessions[session_id]
    
    # Create reports directory if it doesn't exist
    os.makedirs("reports", exist_ok=True)
    
    # Generate Excel report
    excel_path = f"reports/game_report_{session_id}.xlsx"
    
    # Create DataFrames
    metrics_df = pd.DataFrame([
        {
            "verse_number": i + 1,
            "response_time": m.response_time,
            "verse_similarity": m.verse_similarity,
            "context_similarity": m.context_similarity
        }
        for i, m in enumerate(session.performance_history)
    ])
    
    summary_df = pd.DataFrame([{
        "Session ID": session_id,
        "Difficulty": session.difficulty,
        "Total Verses": session.user_verses_count,
        "Strikes": session.strikes,
        "Game Result": session.game_result or "In Progress",
        "Average Response Time": metrics_df['response_time'].mean(),
        "Average Verse Similarity": metrics_df['verse_similarity'].mean(),
        "Average Context Similarity": metrics_df['context_similarity'].mean()
    }])
    
    verses_df = pd.DataFrame({
        "Verse Number": range(1, len(session.conversation_history) + 1),
        "Verse Text": session.conversation_history
    })
    
    # Write to Excel with proper formatting
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Write DataFrames
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        metrics_df.to_excel(writer, sheet_name='Detailed Metrics', index=False)
        verses_df.to_excel(writer, sheet_name='Verses', index=False)
        
        # Get workbook and add formatting
        workbook = writer.book
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#D3D3D3',
            'border': 1
        })
        
        # Format each worksheet
        for sheet_name in ['Summary', 'Detailed Metrics', 'Verses']:
            worksheet = writer.sheets[sheet_name]
            
            # Get the appropriate DataFrame for column widths
            if sheet_name == 'Summary':
                df = summary_df
            elif sheet_name == 'Detailed Metrics':
                df = metrics_df
            else:
                df = verses_df
                
            # Format headers and adjust column widths
            for idx, col in enumerate(df.columns):
                worksheet.write(0, idx, col, header_format)
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(str(col))
                )
                worksheet.set_column(idx, idx, max_length + 2)
    
    # Generate graphs
    graphs = []
    
    # Response Time Graph
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['verse_number'], metrics_df['response_time'], marker='o')
    plt.title('Response Time per Verse')
    plt.xlabel('Verse Number')
    plt.ylabel('Response Time (seconds)')
    plt.grid(True)
    time_graph_path = f"reports/response_time_{session_id}.png"
    plt.savefig(time_graph_path)
    plt.close()
    graphs.append(time_graph_path)
    
    # Similarities Graph
    plt.figure(figsize=(12, 6))
    plt.plot(metrics_df['verse_number'], metrics_df['verse_similarity'], marker='o', label='Verse Similarity')
    plt.plot(metrics_df['verse_number'], metrics_df['context_similarity'], marker='s', label='Context Similarity')
    plt.title('Verse and Context Similarities')
    plt.xlabel('Verse Number')
    plt.ylabel('Similarity Score')
    plt.legend()
    plt.grid(True)
    similarities_graph_path = f"reports/similarities_{session_id}.png"
    plt.savefig(similarities_graph_path)
    plt.close()
    graphs.append(similarities_graph_path)
    
    # Correlation Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics_df[['response_time', 'verse_similarity', 'context_similarity']].corr(), 
                annot=True, cmap='coolwarm')
    plt.title('Metrics Correlation')
    correlation_graph_path = f"reports/correlation_{session_id}.png"
    plt.savefig(correlation_graph_path)
    plt.close()
    graphs.append(correlation_graph_path)
    
    # Calculate summary statistics
    summary_stats = {
        "total_verses": len(session.conversation_history),
        "average_metrics": {
            "response_time": float(metrics_df['response_time'].mean()),
            "verse_similarity": float(metrics_df['verse_similarity'].mean()),
            "context_similarity": float(metrics_df['context_similarity'].mean())
        },
        "performance_rating": calculate_performance_rating(metrics_df)
    }
    
    return AnalyticsReport(
        excel_path=excel_path,
        graphs_paths=graphs,
        summary=summary_stats
    )

@app.get("/analytics/download/{session_id}")
async def download_report(session_id: str):
    """Download the Excel report for a session."""
    excel_path = f"reports/game_report_{session_id}.xlsx"
    if not os.path.exists(excel_path):
        raise HTTPException(status_code=404, detail="Report not found. Generate it first.")
    
    return FileResponse(
        path=excel_path,
        filename=f"poetry_game_report_{session_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
@app.get("/analytics/graphs/{session_id}/{graph_name}")
async def get_graph(session_id: str, graph_name: str):
    """Retrieve a specific graph for a session."""
    graph_path = f"reports/{graph_name}_{session_id}.png"
    if not os.path.exists(graph_path):
        raise HTTPException(status_code=404, detail="Graph not found. Generate report first.")
    
    return FileResponse(graph_path)
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
        current_verses=session.user_verses_count,
        strikes=session.strikes,
        last_letter=session.last_letter or "",
        ai_response="",
        game_over=session.is_game_over,
        game_result=session.game_result or ""
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

if __name__ == "__main__":
    import uvicorn
    
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            workers=1,
            # loop="uvloop",
            limit_concurrency=10,
            limit_max_requests=1000,
            timeout_keep_alive=30,
            log_level="info"
        )
    except Exception as e:
        print(f"Application error: {str(e)}")
        sys.exit(1)