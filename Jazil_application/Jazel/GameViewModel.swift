////
////  GameViewModel.swift
////  gameday
////
////  Created by Muneera Y on 26/04/1446 AH.
////
//
//import Foundation
//import Combine
//
//class GameViewModel: ObservableObject {
//    @Published var gameSession: GameSession?
//    @Published var chatMessages: [ChatMessage] = []
//    @Published var userInput: String = ""
//    @Published var selectedDifficulty: Difficulty = .easy // Default difficulty
//    @Published var errorMessage: String?
//    private let baseURL = "https://jazil-game-53460647872.us-central1.run.app/games"
//    var cancellable: AnyCancellable?
//    
//    // New properties to track history
//    @Published var userMoves: [String] = []       // Track user moves
//    @Published var apiResponses: [String] = []    // Track API responses
//
//    func getInitialMessage(completion: @escaping (String) -> Void) {
//        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
//            completion("Welcome to the game! Let's start.")
//        }
//    }
//    
//    // Function to create a new game session with selected difficulty
//    func createGameSession(difficulty: Difficulty) {
//        let url = URL(string: baseURL)!
//        var request = URLRequest(url: url)
//        request.httpMethod = "POST"
//        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
//        
//        // Include difficulty in the request body
//        let body: [String: Any] = ["difficulty": difficulty.rawValue]
//        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
//
//        cancellable = URLSession.shared.dataTaskPublisher(for: request)
//            .map { $0.data }
//            .decode(type: GameSession.self, decoder: JSONDecoder())
//            .receive(on: DispatchQueue.main)
//            .sink(receiveCompletion: { completion in
//                if case .failure(let error) = completion {
//                    self.errorMessage = "Failed to create game: \(error.localizedDescription)"
//                }
//            }, receiveValue: { [weak self] session in
//                // Store the session data
//                self?.gameSession = session
//
//                // Check AI response
//                if let aiResponse = session.ai_response, !aiResponse.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
//                    // If valid AI response, append it to apiResponses
//                    self?.apiResponses.append("api: \(aiResponse)")
//                    
//                    // Create AI ChatMessage and add to chatMessages
//                    let aiChat = ChatMessage(message: aiResponse, isUser: false)
//                    self?.chatMessages.append(aiChat)
//                } else {
//                    // If AI response is empty, log a default message
//                    let defaultMessage = "No AI response received."
//                    self?.apiResponses.append(defaultMessage)
//                    let aiChat = ChatMessage(message: "اهلا بكم في جازل ", isUser: false)
//                    self?.chatMessages.append(aiChat)
//                }
//
//                // Add the initial user move (e.g., session message)
//                if let userMessage = self?.gameSession?.message {
//                    self?.userMoves.append(userMessage)
//                }
//            })
//    }
//
//
//
//
//    func submitMove(userMove: String) {
//        guard let sessionID = gameSession?.session_id else {
//            errorMessage = "No active game session."
//            return
//        }
//
//        let url = URL(string: "\(baseURL)/\(sessionID)/verses")!
//        var request = URLRequest(url: url)
//        request.httpMethod = "POST"
//        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
//        
//        let body: [String: Any] = ["verse": userMove]
//        request.httpBody = try? JSONSerialization.data(withJSONObject: body)
//
//        // Log the body and URL for debugging
//        print("Submitting move to URL: \(url)")
//        print("Request body: \(body)")
//        
//        cancellable = URLSession.shared.dataTaskPublisher(for: request)
//            .map { $0.data }
//            .decode(type: GameSession.self, decoder: JSONDecoder())
//            .receive(on: DispatchQueue.main)
//            .sink(receiveCompletion: { completion in
//                if case .failure(let error) = completion {
//                    self.errorMessage = "Failed to submit move: \(error.localizedDescription)"
//                    print("Error: \(error.localizedDescription)") // Log the error
//                }
//            }, receiveValue: { [weak self] updatedSession in
//                if let aiResponse = updatedSession.ai_response {
//                    if aiResponse.count <= 0{
//                        print(" \(aiResponse) اجابة علام :") // Log the response
//                        self?.apiResponses.append("api: \(aiResponse)............................ message: \(updatedSession.message) .............................. last letter: \(String(describing: updatedSession.last_letter)) :")
//                        
//                        // Add the AI response to chatMessages as a non-user message
//                        let aiChat = ChatMessage(message: updatedSession.message, isUser: false)
//                        self?.chatMessages.append(aiChat)
//                    }else{
//                        self?.apiResponses.append("api: \(aiResponse)............................ message: \(updatedSession.message) .............................. last letter: \(String(describing: updatedSession.last_letter)) :")
//                        let aiChat = ChatMessage(message: aiResponse, isUser: false)
//                    }
//                    
//                } else {
//                    print("No AI response available.")
//                    self?.apiResponses.append("No response received.")
//                }
//            })
//    }
//
//
//}
//

import Foundation
import Combine

class GameViewModel: ObservableObject {
    @Published var gameSession: GameSession?
    @Published var chatMessages: [ChatMessage] = []
    @Published var userInput: String = ""
    @Published var selectedDifficulty: Difficulty = .easy // Default difficulty
    @Published var errorMessage: String?
    @Published var strick: Int = 0
    @Published var isVerse: Bool = false
    @Published var game_result_final: String = ""
    
    private let baseURL = "https://jazil-game-53460647872.us-central1.run.app/games"
    
    var cancellable: AnyCancellable?
    // New properties to track history
    @Published var userMoves: [String] = []       // Track user moves
    @Published var apiResponses: [String] = []    // Track API responses

    func getInitialMessage(completion: @escaping (String) -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            completion("Welcome to the game! Let's start.")
        }
    }
    
    // Function to create a new game session with selected difficulty
    func createGameSession(difficulty: Difficulty) {
        guard let url = URL(string: baseURL) else {
            errorMessage = "Invalid URL"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = ["difficulty": difficulty.rawValue]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        cancellable = URLSession.shared.dataTaskPublisher(for: request)
            .map { $0.data }
            .decode(type: GameSession.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    self.errorMessage = "Failed to create game: \(error.localizedDescription)"
                }
            }, receiveValue: { [weak self] session in
                self?.gameSession = session
                self?.handleAIResponse(session: session)
                self?.game_result_final = session.game_result
            })
    }

    // Function to submit a user move
    func submitMove(userMove: String) {
        guard let sessionID = gameSession?.session_id else {
            errorMessage = "No active game session."
            return
        }

        guard let url = URL(string: "\(baseURL)/\(sessionID)/verses") else {
            errorMessage = "Invalid URL"
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let body: [String: Any] = ["verse": userMove]
        request.httpBody = try? JSONSerialization.data(withJSONObject: body)

        print("Submitting move to URL: \(url)")  // Log URL for debugging
        print("Request body: \(body)")          // Log body for debugging

        cancellable = URLSession.shared.dataTaskPublisher(for: request)
            .map { $0.data }
            .decode(type: GameSession.self, decoder: JSONDecoder())
            .receive(on: DispatchQueue.main)
            .sink(receiveCompletion: { completion in
                if case .failure(let error) = completion {
                    self.errorMessage = "Failed to submit move: \(error.localizedDescription)"
                    print("Error: \(error.localizedDescription)")
                }
            }, receiveValue: { [weak self] updatedSession in
                
                self?.handleAIResponse(session: updatedSession)
                self?.strick = updatedSession.strikes
                print(self?.strick)
                self?.game_result_final = updatedSession.game_result
                self?.userMoves.append(userMove)
            })
       
    }
    
    // Helper function to handle AI response from the game session
    private func handleAIResponse(session: GameSession) {
        guard let aiResponse = session.ai_response else {
            print("No AI response available.")
            apiResponses.append("No response received.")
            return
        }
        
        if aiResponse.isEmpty {
            print("Empty AI response received: \(aiResponse)")
//            apiResponses.append("API: Empty response.")
            print ("here in the view model ",session.strikes)
            strick = session.strikes
        
            let aiChat = ChatMessage(message: session.message, isUser: false)
            chatMessages.append(aiChat)
            if session.current_verses > 1{
                isVerse = true
            }else{
                isVerse = false
            }
        } else {
            print("AI response: \(aiResponse)")
            apiResponses.append("API: \(aiResponse)")
            strick = session.strikes
            // Create AI ChatMessage and add to chatMessages
            let aiChat = ChatMessage(message: aiResponse, isUser: false)
            chatMessages.append(aiChat)
        }

        // Directly append the session message since it's not optional
        userMoves.append(session.message)
    }

}
