import SwiftUI

struct GameView: View {
    @StateObject var viewModel = GameViewModel()
    var selectedDifficulty: Difficulty // Pass the difficulty here

    var body: some View {
        VStack {
            Text("Selected Difficulty: \(selectedDifficulty.rawValue.capitalized)")
                .font(.headline)
                .padding()
            
            // Game mechanics and UI as per your existing GameView
            if let session = viewModel.gameSession {
                Text("Game ID: \(session.session_id)")
                Text(session.message)
                Text("Verses: \(session.current_verses)/\(session.required_verses)")
                Text("Strikes: \(session.strikes)")
                
                if let lastLetter = session.last_letter {
                    Text("Last Letter: \(lastLetter)")
                }
                
                if let aiResponse = session.ai_response {
                    Text("AI Response: \(aiResponse)")
                }
            } else {
                Text("No active game session")
            }
            
            HStack {
                TextField("Enter your move", text: $viewModel.userInput)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()

                Button("Submit") {
                    viewModel.submitMove(userMove: viewModel.userInput)
                    viewModel.userInput = "" // Clear input after submission
                }
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(10)
            }
            .padding()

            if let error = viewModel.errorMessage {
                Text(error).foregroundColor(.red)
            }

            Spacer()
            
            // Display User Moves
            Text("User Moves:")
                .font(.headline)
                .padding(.top)
            List(viewModel.apiResponses, id: \.self) { move in
                Text(move)
            }
            .frame(height: 100) // Limit the height for user moves

            // Display API Responses
            Text("API Responses:")
                .font(.headline)
                .padding(.top)
            List(viewModel.apiResponses, id: \.self) { response in
                Text(response)
            }
            .frame(height: 100) // Limit the height for API responses
        }
        .padding()
        .onAppear {
            viewModel.createGameSession(difficulty: selectedDifficulty)
            print("Attempted to create a new game session with difficulty: \(selectedDifficulty.rawValue)")
        }
    }
}

#Preview {
    GameView(selectedDifficulty: .easy)
}
