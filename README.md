# Project Jazil - Arabic Poetry Game

## Overview
Project Jazil is an innovative Arabic poetry game that combines classical Arabic poetry with modern AI technology. The system consists of a powerful backend API and a native iOS application (Jazel), enabling interactive verse exchange between users and AI.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [iOS Application](#ios-application)
- [Analytics](#analytics)
- [Contributing](#contributing)
- [License](#license)

## Features

### Game Mechanics
- Turn-based poetry exchange
- Real-time verse validation
- Classical Arabic poetry rules enforcement
- Multiple difficulty levels
- Performance tracking and analytics

### Technical Features
- FastAPI backend with async support
- IBM Watson AI integration
- Comprehensive analytics system
- Native iOS application (Jazel)
- Real-time performance metrics
- Automated report generation

## System Architecture

### Project Structure
```
Project-Jazil/
├── main_game_backend/        # Game server and API
├── Jazil_application/        # iOS client application
│   ├── Jazel/               # iOS app source code
│   └── Jazel.xcodeproj/     # Xcode project files
├── data/                     # Data storage (ignored in git)
├── reports/                  # Generated reports (ignored in git)
└── docs/                     # Documentation
```

### Components
1. **Backend Server**
   - FastAPI framework
   - IBM Watson integration
   - Poetry validation system
   - Analytics engine

2. **iOS Application (Jazel)**
   - Native Swift implementation
   - Real-time gameplay interface
   - Performance visualization
   - Arabic text support
   - Localization support

3. **Analytics System**
   - Real-time metrics tracking
   - Report generation
   - Performance visualization

## Installation

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/Project-Jazil.git

# Navigate to backend directory
cd Project-Jazil/main_game_backend

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your credentials

# Run the server
uvicorn jazil_game_server:app --host 0.0.0.0 --port 8000
```

### iOS Application Setup
1. Prerequisites:
   - Xcode 15.0+
   - iOS 15.0+ deployment target
   - macOS Sonoma 14.0+ for development

2. Installation Steps:
```bash
# Navigate to iOS project
cd Project-Jazil/Jazil_application

# Open Xcode project
open Jazel.xcodeproj

# Build and run the application
# Select your target device/simulator in Xcode and click Run
```

### Environment Variables
```env
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_PROJECT_ID=your_project_id
IBM_WATSONX_URL=your_api_url
```

## Usage

### API Endpoints
```http
# Create new game
POST /games

# Submit verse
POST /games/{session_id}/verses

# Get game metrics
GET /games/{session_id}/metrics

# Generate analytics report
GET /analytics/report/{session_id}
```

### iOS Application Features
1. User Authentication
   - Sign up/Sign in
   - Profile management

2. Game Features
   - Create new games
   - Multiple difficulty levels
   - Real-time verse submission
   - Performance tracking

3. Analytics
   - In-app performance metrics
   - Historical game data
   - Progress tracking

## API Documentation

### Game Creation
```json
POST /games
{
    "difficulty": "easy"
}
```

### Verse Submission
```json
POST /games/{session_id}/verses
{
    "verse": "كُن اِبنَ مَن شِئتَ واِكتَسِب أَدَباً
يُغنيكَ مَحمُودُهُ عَنِ النَسَبِ"
}
```

### Response Format
```json
{
    "session_id": "uuid",
    "message": "Success message",
    "performance_metrics": {
        "response_time": 2.5,
        "verse_similarity": 0.85,
        "context_similarity": 0.75
    }
}
```

## Analytics

### Available Metrics
- Response time
- Verse similarity
- Context similarity
- Overall performance rating

### Report Types
- Excel reports with detailed analysis
- Performance graphs
- Correlation heatmaps
- Statistical summaries

## Contributing

### Backend Development
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### iOS Development
1. Fork the repository
2. Open Jazel.xcodeproj
3. Create a feature branch
4. Make changes following Swift style guidelines
5. Submit a Pull Request

### Development Requirements
- Python 3.8+ (Backend)
- FastAPI (Backend)
- IBM Watson API access
- Xcode 15.0+ (iOS)
- Swift 5.0+ (iOS)
- iOS 15.0+ deployment target

## Support

### Platform Requirements
- iOS 15.0 or later
- Compatible with iPhone and iPad
- Arabic language support required

### Technical Support
For technical issues or questions:
- Backend: Create an issue in the GitHub repository
- iOS: Check Jazel documentation or create an issue


