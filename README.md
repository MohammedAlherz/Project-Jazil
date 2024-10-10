# Jazil Game Documentation

## Table of Contents

1. Introduction
   1.1 Purpose
   1.2 Game Overview
   1.3 Target Audience

2. System Requirements
   2.1 Hardware Requirements
   2.2 Software Requirements
   2.3 Dependencies

3. Installation and Setup
   3.1 Installing Required Libraries
   3.2 Setting Up Environment Variables
   3.3 Preparing the Dataset

4. Game Architecture
   4.1 High-Level Overview
   4.2 Main Components
   4.3 Data Flow

5. Code Structure
   5.1 Main Functions
   5.2 Helper Functions
   5.3 Game Loop

6. Key Technologies
   6.1 Natural Language Processing
   6.2 Embeddings
   6.3 IBM Watson API

7. Game Flow
   7.1 Starting the Game
   7.2 User Input Handling
   7.3 Verse Validation
   7.4 AI Response Generation
   7.5 Ending the Game

8. Customization and Extension
   8.1 Modifying Validation Criteria
   8.2 Adjusting AI Response Generation
   8.3 Extending the Dataset

9. Troubleshooting
   9.1 Common Issues
   9.2 Error Messages
   9.3 Support Resources

10. Conclusion
    10.1 Summary
    10.2 Future Enhancements

11. Appendices
    11.1 Glossary of Terms
    11.2 References
    11.3 Version History

---

## 1. Introduction

### 1.1 Purpose

This documentation provides a comprehensive guide to the Jazil game, an interactive poetry exchange system that utilizes artificial intelligence to engage users in classical Arabic poetry composition.

### 1.2 Game Overview

Jazil is a turn-based game where users and an AI take turns composing lines of classical Arabic poetry. The game validates user inputs for poetic structure and generates AI responses that adhere to traditional Arabic poetry rules.

### 1.3 Target Audience

This documentation is intended for developers, researchers, and poetry enthusiasts interested in understanding, deploying, or extending the Jazil game system.

## 2. System Requirements

### 2.1 Hardware Requirements

- A computer with at least 4GB of RAM
- 100MB of free disk space

### 2.2 Software Requirements

- Python 3.7 or higher
- Internet connection for API access

### 2.3 Dependencies

- pandas
- numpy
- requests
- python-dotenv
- langchain
- scikit-learn
- sentence-transformers

## 3. Installation and Setup

### 3.1 Installing Required Libraries

Install the required Python libraries using pip:

```
pip install pandas numpy requests python-dotenv langchain scikit-learn sentence-transformers
```

### 3.2 Setting Up Environment Variables

Create a `.env` file in the project root directory with the following content:

```
IBM_WATSONX_API_KEY=your_api_key
IBM_WATSONX_PROJECT_ID=your_project_id
IBM_WATSONX_URL=your_api_url
```

Replace `your_api_key`, `your_project_id`, and `your_api_url` with your actual IBM Watson credentials.

### 3.3 Preparing the Dataset

Place the "Arabic Poem Comprehensive Dataset (APCD).csv" file in the project root directory.

## 4. Game Architecture

### 4.1 High-Level Overview

Jazil uses a combination of natural language processing techniques, including embeddings and API-based language models, to create an interactive poetry exchange system.

### 4.2 Main Components

1. Data Preprocessing
2. Embedding Generation
3. Verse Validation
4. AI Response Generation
5. Game Loop

### 4.3 Data Flow

1. User input → Verse Validation → Game History
2. Game History → Embedding Comparison → Similar Verses
3. Similar Verses + Game History → AI Response Generation → Game History

## 5. Code Structure

### 5.1 Main Functions

- `play_jazil()`: Main game loop
- `generate_response()`: Generates AI responses
- `retrieve_similar_verses()`: Finds semantically similar verses

### 5.2 Helper Functions

- `get_access_token()`: Retrieves IBM Watson API token
- `load_and_clean_data()`: Preprocesses the dataset
- `create_embeddings()`: Generates or loads verse embeddings
- `normalize_letter()`: Normalizes Arabic letters
- `starts_with_letter()`: Checks verse starting letter
- `get_last_letter()`: Extracts the last letter of a verse

### 5.3 Game Loop

The `play_jazil()` function contains the main game loop, handling user input, verse validation, and AI response generation.

## 6. Key Technologies

### 6.1 Natural Language Processing

Jazil employs various NLP techniques for text processing, including tokenization, normalization, and semantic similarity comparison.

### 6.2 Embeddings

Sentence embeddings are generated using the SentenceTransformer model "all-MiniLM-L6-v2" to capture semantic meanings of verses.

### 6.3 IBM Watson API

The IBM Watson API is used for verse validation and AI response generation, leveraging advanced language models for poetry analysis and creation.

## 7. Game Flow

### 7.1 Starting the Game

The game begins by loading the dataset, creating embeddings, and welcoming the user.

### 7.2 User Input Handling

Users input their verses, which are then checked for uniqueness and adherence to the last-letter rule.

### 7.3 Verse Validation

User inputs are validated using the IBM Watson API, checking for poetic structure, meter, rhyme, and authenticity.

### 7.4 AI Response Generation

The AI generates responses based on the conversation context, similar verses from the dataset, and classical Arabic poetry rules.

### 7.5 Ending the Game

The game continues until the user decides to quit by typing 'quit'.

## 8. Customization and Extension

### 8.1 Modifying Validation Criteria

Adjust the validation prompt in the `play_jazil()` function to change verse acceptance criteria.

### 8.2 Adjusting AI Response Generation

Modify the AI response generation prompt to alter the style or rules of generated verses.

### 8.3 Extending the Dataset

Expand the "Arabic Poem Comprehensive Dataset (APCD).csv" file with additional poems to increase the game's vocabulary and style range.

## 9. Troubleshooting

### 9.1 Common Issues

- API connection errors
- Dataset loading failures
- Embedding generation issues

### 9.2 Error Messages

Detailed error messages are printed to the console for debugging purposes.

### 9.3 Support Resources

For additional support, refer to the documentation of the used libraries and the IBM Watson API.

## 10. Conclusion

### 10.1 Summary

Jazil demonstrates the potential of AI in interactive poetry generation and analysis, providing a unique platform for engaging with classical Arabic poetry.

### 10.2 Future Enhancements

Potential improvements include multi-player support, integration with voice recognition, and expansion to other poetic traditions.

## 11. Appendices

### 11.1 Glossary of Terms

- Embedding: A numerical representation of text that captures semantic meaning
- API: Application Programming Interface
- NLP: Natural Language Processing

### 11.2 References

- SentenceTransformers documentation
- IBM Watson API documentation
- Classical Arabic Poetry resources

### 11.3 Version History

- v1.0: Initial release
- v1.1: Added embedding caching for improved performance

