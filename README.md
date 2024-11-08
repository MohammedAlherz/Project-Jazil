# Project Jazil - Arabic Poetry Game

## Overview
Project Jazil is an innovative Arabic poetry game combining classical Arabic poetry with modern AI technology. Through a powerful backend API and native iOS application (Jazel), users can engage in interactive verse exchanges with an AI system, creating a unique educational and entertaining experience.

## Key Features

### Game Experience
- Interactive turn-based poetry exchange
- Real-time verse validation using classical Arabic rules
- Multiple difficulty levels for different skill sets
- Performance tracking and detailed analytics
- Educational insights into classical Arabic poetry

### Technical Capabilities
- FastAPI-powered backend with IBM Watson AI integration
- Native iOS application (Jazel) for seamless mobile experience
- Real-time performance metrics and analytics
- Automated reporting system

## System Components

### Project Structure
```
Project-Jazil/
├── main_game_backend/        # Game server and API
├── Jazil_application/        # iOS client application
│   ├── Jazel/               # iOS app source code
│   └── Jazel.xcodeproj/     # Xcode project files
├── data/                     # Data storage
└── reports/                  # Generated reports
```

### Server Components
- FastAPI framework
- IBM Watson AI integration
- Poetry validation engine
- Analytics system

### Mobile Application
- Native iOS implementation
- Arabic text and localization support
- Real-time gameplay interface
- Performance visualization

## Setup Requirements

### Backend
- Python 3.8+
- FastAPI
- IBM Watson API credentials
- Required Python packages (see requirements.txt)

### iOS Application
- Xcode 15.0+
- iOS 15.0+ deployment target
- macOS Sonoma 14.0+ for development

## API Features

### Main Endpoints
```http
POST /games                        # Create game
POST /games/{session_id}/verses    # Submit verse
GET /games/{session_id}/metrics    # Get metrics
GET /analytics/report/{session_id}  # Generate report
```

### Example Game Flow
```json
// Create Game
POST /games
{
    "difficulty": "easy"
}

// Submit Verse
POST /games/{session_id}/verses
{
    "verse": "كُن اِبنَ مَن شِئتَ واِكتَسِب أَدَباً
يُغنيكَ مَحمُودُهُ عَنِ النَسَبِ"
}
```

## Analytics Features
- Response time tracking
- Verse similarity analysis
- Context similarity measurement
- Performance reporting
- Visual analytics

## Platform Support
- iOS 15.0+
- iPhone and iPad compatibility
- Arabic language support
- Internet connection required

For technical support or questions, please create an issue in the GitHub repository.

