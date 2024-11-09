# Jazil Game Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation and Setup](#3-installation-and-setup)
4. [Game Architecture](#4-game-architecture)
5. [AI Models and Dataset](#5-ai-models-and-dataset)
6. [API Endpoints](#6-api-endpoints)
7. [Analytics and Reporting](#7-analytics-and-reporting)
8. [Troubleshooting](#8-troubleshooting)
9. [Security and Performance](#9-security-and-performance)

## 1. Introduction

### 1.1 Purpose
Jazil is an AI-powered Arabic poetry game that enables interactive verse exchange between users and an AI system, featuring real-time performance analytics and reporting.

### 1.2 Game Overview
- Turn-based poetry exchange
- Real-time verse validation
- Performance metrics tracking
- Comprehensive analytics
- Multiple difficulty levels
- Interactive API endpoints
- Arabic poetry rules enforcement
- Semantic similarity matching

### 1.3 Target Audience
- Developers
- Poetry enthusiasts
- Researchers
- Educational institutions
- Arabic language learners
- Cultural institutions

## 2. System Requirements

### 2.1 Hardware Requirements
- RAM: Minimum 8GB (16GB recommended)
- Storage: 1GB free space for application, 2GB for embeddings
- Processor: Modern multi-core CPU
- Internet connection required (5 Mbps minimum)
- GPU: Optional but recommended for faster embedding generation

### 2.2 Software Requirements
- Python 3.8 or higher
- pip package manager
- Git (optional)
- Node.js (for development tools)
- Docker (optional, for containerization)

### 2.3 Dependencies
```plaintext
fastapi>=0.68.0
uvicorn>=0.15.0
pydantic>=1.8.2
requests>=2.26.0
pandas>=1.3.0
numpy>=1.21.0
python-dotenv>=0.19.0
datasets>=2.0.0
sentence-transformers>=2.2.0
langchain>=0.0.200
langchain_community>=0.0.10
scikit-learn>=0.24.2
transformers>=4.21.0
sentencepiece>=0.1.96
camel-tools>=1.0.0
apscheduler>=3.9.1
torch>=1.9.0
h5py>=3.6.0
tqdm>=4.62.2
psutil>=5.8.0
pyarrow>=6.0.0
uvloop>=0.16.0
matplotlib>=3.4.3
seaborn>=0.11.2
xlsxwriter>=3.0.2
```

## 3. Installation and Setup

### 3.1 Environment Setup
```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Initialize embeddings (first run only)
python init_embeddings.py
```

### 3.2 Environment Variables
Create a `.env` file:
```env
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_PROJECT_ID=your_project_id
IBM_WATSONX_URL=your_api_url
MODEL_ID=sdaia/allam-1-13b-instruct
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_SESSIONS=1000
CLEANUP_INTERVAL=900
DEBUG=False
```

### 3.3 Running the Server
```bash
# Development
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4 --limit-concurrency 50
```

## 4. Game Architecture

### 4.1 Core Components
- FastAPI web server
- IBM Watson integration
- Embedding system
- Analytics engine
- Report generation
- Session manager
- Verse validator
- Performance tracker

### 4.2 Key Features
- Real-time verse validation
- Performance metrics tracking
- Session management
- Automatic cleanup
- Excel report generation
- Visual analytics
- Semantic similarity analysis
- Arabic poetry rules enforcement

## 5. AI Models and Dataset

### 5.1 Language Models
#### Allam-1 Model (IBM WatsonX)
- Model ID: sdaia/allam-1-13b-instruct
- Purpose: Verse generation and validation
- Parameters: 13 billion
- Specialization: Arabic poetry and literature
- Integration: IBM WatsonX platform

#### Embedding Model
- Model: all-MiniLM-L6-v2
- Source: Hugging Face Sentence Transformers
- Vector size: 384 dimensions
- Purpose: Semantic similarity computation
- Performance: Optimized for multilingual content

### 5.2 Dataset
- Source: arbml/ashaar (Hugging Face)
- Content: Classical Arabic poetry collection
- Storage format: Parquet + HDF5
- Embeddings: Pre-computed and cached
- Updates: Manual refresh available

## 6. API Endpoints

### 6.1 Game Management
```http
POST /games
Request:
{
    "difficulty": "easy" | "hard"
}
Response:
{
    "session_id": "uuid",
    "message": "string",
    "required_verses": int,
    "current_verses": 0,
    "strikes": 0,
    "last_letter": "",
    "ai_response": "",
    "game_over": false,
    "game_result": ""
}

GET /games/{session_id}
DELETE /games/{session_id}
```

### 6.2 Gameplay
```http
POST /games/{session_id}/verses
Request:
{
    "verse": "string"
}
Response:
{
    "session_id": "uuid",
    "message": "string",
    "required_verses": int,
    "current_verses": int,
    "strikes": int,
    "last_letter": "string",
    "ai_response": "string",
    "game_over": boolean,
    "game_result": "string",
    "performance_metrics": {
        "response_time": float,
        "verse_similarity": float,
        "context_similarity": float
    }
}
```

### 6.3 Analytics
```http
GET /games/{session_id}/metrics
GET /analytics/report/{session_id}
GET /analytics/download/{session_id}
GET /analytics/graphs/{session_id}/{graph_name}
```

## 7. Analytics and Reporting

### 7.1 Performance Metrics
- Response time (seconds)
- Verse similarity (0-1)
- Context similarity (0-1)
- Overall performance rating
- Strike count
- Completion rate
- Session duration

### 7.2 Report Types
- Excel reports with multiple sheets
- Performance graphs
- Correlation heatmaps
- Statistical summaries
- Session analytics
- User progression tracking

### 7.3 Available Graphs
- Response time trends
- Similarity comparisons
- Metric correlations
- Performance distribution
- Learning curve analysis

## 8. Troubleshooting

### 8.1 Common Issues
```plaintext
Error: "Game session not found"
Solution: Ensure session_id is valid and session hasn't expired
Prevention: Implement session refresh mechanism

Error: "Report not found"
Solution: Generate report before downloading
Prevention: Add report status checking

Error: "Graph not found"
Solution: Generate analytics report first
Prevention: Implement automatic report generation

Error: "Model timeout"
Solution: Retry request with exponential backoff
Prevention: Implement request queuing
```

### 8.2 Performance Optimization
- Use uvloop for better async performance
- Implement proper garbage collection
- Monitor memory usage with psutil
- Cache embeddings using h5py
- Implement connection pooling
- Use batch processing where applicable
- Optimize database queries
- Implement rate limiting

## 9. Security and Performance

### 9.1 Security Measures
- API key validation
- Rate limiting
- Session timeout
- Input sanitization
- Error handling
- Secure token management
- Data encryption
- Access control

### 9.2 Performance Monitoring
- Server metrics tracking
- Response time monitoring
- Memory usage tracking
- Model performance metrics
- Session analytics
- Error rate monitoring
- Resource utilization

### 9.3 Maintenance
- Regular embedding updates
- Model performance evaluation
- Database optimization
- Log rotation
- Backup procedures
- Security updates
- Performance tuning
