# Jazil Game Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Requirements](#2-system-requirements)
3. [Installation and Setup](#3-installation-and-setup)
4. [Game Architecture](#4-game-architecture)
5. [API Endpoints](#5-api-endpoints)
6. [Analytics and Reporting](#6-analytics-and-reporting)
7. [Troubleshooting](#7-troubleshooting)

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

### 1.3 Target Audience
- Developers
- Poetry enthusiasts
- Researchers
- Educational institutions

## 2. System Requirements

### 2.1 Hardware Requirements
- RAM: Minimum 8GB (16GB recommended)
- Storage: 1GB free space
- Processor: Modern multi-core CPU
- Internet connection required

### 2.2 Software Requirements
- Python 3.8 or higher
- pip package manager
- Git (optional)

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
matplotlib
seaborn
xlsxwriter
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
```

### 3.2 Environment Variables
Create a `.env` file:
```env
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_PROJECT_ID=your_project_id
IBM_WATSONX_URL=your_api_url
```

### 3.3 Running the Server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 4. Game Architecture

### 4.1 Core Components
- FastAPI web server
- IBM Watson integration
- Embedding system
- Analytics engine
- Report generation

### 4.2 Key Features
- Real-time verse validation
- Performance metrics tracking
- Session management
- Automatic cleanup
- Excel report generation
- Visual analytics

## 5. API Endpoints

### 5.1 Game Management
```http
POST /games
GET /games/{session_id}
DELETE /games/{session_id}
```

### 5.2 Gameplay
```http
POST /games/{session_id}/verses
```

### 5.3 Analytics
```http
GET /games/{session_id}/metrics
GET /analytics/report/{session_id}
GET /analytics/download/{session_id}
GET /analytics/graphs/{session_id}/{graph_name}
```

### 5.4 System
```http
GET /health
```

## 6. Analytics and Reporting

### 6.1 Performance Metrics
- Response time
- Verse similarity
- Context similarity
- Overall performance rating

### 6.2 Report Types
- Excel reports with multiple sheets
- Performance graphs
- Correlation heatmaps
- Statistical summaries

### 6.3 Available Graphs
- Response time trends
- Similarity comparisons
- Metric correlations

## 7. Troubleshooting

### 7.1 Common Issues
```plaintext
Error: "Game session not found"
Solution: Ensure session_id is valid and session hasn't expired

Error: "Report not found"
Solution: Generate report before downloading

Error: "Graph not found"
Solution: Generate analytics report first
```

### 7.2 Performance Optimization
- Use uvloop for better async performance
- Implement proper garbage collection
- Monitor memory usage with psutil
- Cache embeddings using h5py

### 7.3 Support Resources
- IBM Watson documentation
- FastAPI documentation
- Project GitHub repository
- Issue tracker

## Development and Testing

### Local Development
```bash
# Run with reload for development
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing API Endpoints
Use Postman or curl:
```bash
# Health check
curl http://localhost:8000/health

# Create game
curl -X POST http://localhost:8000/games \
     -H "Content-Type: application/json" \
     -d '{"difficulty": "easy"}'
```

## Data Management

### Embeddings Storage
```plaintext
/data/
  ├── embeddings.h5
  └── verses.parquet
```

