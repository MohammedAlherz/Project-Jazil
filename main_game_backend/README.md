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
10. [Cloud Deployment](#10-cloud-deployment)

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
- Docker (for containerization)
### 2.3 Dependencies
```plaintext
fastapi
uvicorn
pydantic
requests
pandas
numpy
python-dotenv
datasets
sentence-transformers
langchain
langchain_community
scikit-learn
transformers
sentencepiece
camel-tools
apscheduler
torch
h5py
tqdm
psutil
pyarrow 
uvloop  
```

## 3. Installation and Setup

### 3.1 Local Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Environment Variables
Create a `.env` file:
```env
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_PROJECT_ID=your_project_id
IBM_WATSONX_URL=your_api_url
GOOGLE_CLOUD_PROJECT=your_project_id
```

### 3.3 Deployment Options

#### Local Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Google Cloud Run Deployment

1. **Dockerfile Configuration**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PORT 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

2. **Build and Deploy**
```bash
# Build container
gcloud builds submit --tag gcr.io/[PROJECT_ID]/jazil-game

# Deploy to Cloud Run
gcloud run deploy jazil-game \
  --image gcr.io/[PROJECT_ID]/jazil-game \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2
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
- Configuration:
  - Max tokens: 200
  - Decoding: Greedy
  - Repetition penalty: 1.0

#### Embedding Model
- Model: all-MiniLM-L6-v2
- Source: Hugging Face Sentence Transformers
- Vector size: 384 dimensions
- Purpose: Semantic similarity computation
- Batch processing: 100 verses
- Memory optimization: Garbage collection

### 5.2 Dataset
- Source: arbml/ashaar (Hugging Face)
- Content: Classical Arabic poetry collection
- Storage format: 
  - Embeddings: HDF5 with compression
  - Verses: Parquet
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
- Use batch processing
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

## 10. Cloud Run Configuration

### 10.1 Service Configuration
```yaml
service: jazil-game
region: us-central1
platform: managed

runtime_config:
  python_version: "3.9"
  operating_system: "linux"

resources:
  cpu: 2
  memory: "2Gi"
  startup_cpu_boost: true

scaling:
  min_instances: 0
  max_instances: 100
  target_cpu_utilization: 0.65
  request_timeout: 300s
  
concurrency:
  max_instances: 80
  target: 50

vpc_connector: "projects/[PROJECT_ID]/locations/[REGION]/connectors/[CONNECTOR]"
```

### 10.2 Cloud Monitoring
- Request latency tracking
- Error rate monitoring
- Instance count metrics
- CPU/Memory utilization
- Cold start frequency
- Custom metrics dashboard

### 10.3 Cloud Logging
- Application logs
- Request tracing
- Error reporting
- Deployment history
- Security audit logs

### 10.4 Maintenance
- Regular embedding updates
- Model performance evaluation
- Database optimization
- Log rotation
- Backup procedures
- Security updates
- Performance tuning

### 10.5 Disaster Recovery
- Automatic instance recovery
- Regional failover
- Data backup strategy
- Version control
- Configuration management
