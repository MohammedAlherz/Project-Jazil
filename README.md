# Project Jazil - Arabic Poetry Game

## Overview
Project Jazil is an innovative Arabic poetry game combining classical Arabic poetry with modern AI technology. Through a powerful backend API and native iOS application (Jazel), users can engage in interactive verse exchanges with an AI system, creating a unique educational and entertaining experience.

### App Launch & Brand
<div align="center">
  <img src="images/jazil_launch.png" width="250" alt="Jazil Logo and Launch Screen"/>
  <img src="images/logo.png" width="250" alt="Jazil Logo"/>
  <p><em>Jazil's distinctive logo and app launch experience</em></p>
</div>

### Main Interface
<div align="center">
  <img src="images/main_menu.png" width="250" alt="Main Menu"/>
  <p><em>Main menu featuring Play, Instructions, and Leaderboard options</em></p>
</div>

## Key Features

### Game Modes & Leaderboard
<div align="center">
  <img src="images/difficulty_levels.png" width="250" alt="Difficulty Selection"/>
  <img src="images/leaderboard.png" width="250" alt="Leaderboard"/>
  <p><em>Left: Choose between Easy (سهل) and Hard (صعب) difficulty levels<br/>
  Right: بواقع الشعر - Competitive poetry leaderboard</em></p>
</div>

### Interactive Gameplay
<div align="center">
  <img src="images/game_chat.png" width="250" alt="Game Interface"/>
  <p><em>Real-time poetry exchange interface with AI</em></p>
</div>

## System Components

The game offers:
- Real-time verse validation
- Arabic poetry rules enforcement
- Performance tracking
- User rankings and scores
- Educational insights into classical Arabic poetry

<div align="center">
  <img src="images/instructions.png" width="250" alt="Game Instructions"/>
  <p><em>Game instructions and Arabic poetry challenge rules</em></p>
</div>

## Technical Capabilities
- FastAPI-powered backend with IBM Watson AI integration
- Native iOS application
- Real-time performance metrics
- Automated reporting system
- Secure user data handling

## Platform Support
- iOS 15.0+
- iPhone and iPad compatibility
- Arabic language support
- Internet connection required

## Technical Documentation

### API Features

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
    "verse": "كُن اِبنَ مَن شِئتَ واِكتَسِب أَدَباً"
}
```

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

For technical support or questions, please create an issue in the GitHub repository.