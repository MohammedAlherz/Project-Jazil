// Jazl
//
// Created by Mohanned Alsahaf on 05/04/1446 AH.
//

import SwiftUI
import Combine

// ChatView is responsible for displaying the chat interface of the game,
// allowing users to send messages, view chat history, and manage game timers.
struct ChatView: View {
    @StateObject var viewModel = GameViewModel()
    var selectedDifficulty: Difficulty

    @State private var timeRemaining: TimeInterval = 60
    @State private var timer: Timer?
    @State private var isRunning: Bool = false
    @State private var navigateToHome: Bool = false
    @State var worngAnswer: Int = 0
    @State private var showFirstX: Bool = false
    @State private var showSecondX: Bool = false
    @State private var showThirdX: Bool = false

    @State private var messageText: String = ""
    
    
    @State private var userMessage: String = ""
    @State private var animate: Bool = false
    
    @State var chatMessages:[ChatMessage] = []
    @State private var cancellables = Set<AnyCancellable>()
    
    
    @State var showAlert:Bool = false

    var body: some View {
        ZStack {
            LinearGradient(
                gradient: Gradient(stops: [
                    .init(color: Color.white.opacity(0.3), location: 0), // 8% opacity at 0% position
                    .init(color: Color(red: 1.0, green: 166/255, blue: 59/255).opacity(0.5), location: 0.42) // FFA63B at 42%
                ]),
                startPoint: .top,
                endPoint: .bottom
            ).edgesIgnoringSafeArea(.all)
            VStack {
                HStack {
                    
                    
                    HStack{
                        if viewModel.strick >= 1 || worngAnswer >= 1 {
                            
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.red)
                                .wiggle(true)
                            
                        }else {
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.gray.opacity(0.3))
                            
                        }
                        
                        
                        if viewModel.strick >= 2 || worngAnswer >= 2{
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.red)
                                .padding(.leading,5)
                                .wiggle(true)
                        }
                        else {
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.gray.opacity(0.3))
                                .padding(.leading,5)
                            
                        }
                        if viewModel.strick >= 3 || worngAnswer >= 3
                        {
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.red)
                                .padding(.leading,5)
                                .wiggle(true)
                            
                            
                            
                            
                        }else {
                            Text("X")
                                .font(.largeTitle)
                                .fontWeight(.heavy)
                                .italic()
                                .foregroundStyle(.gray.opacity(0.3))
                                .padding(.leading,5)
                            
                        }
                    }
                    //
                    
                    .frame(width: 100)
                    // }
                    Spacer()
                    
                    VStack {
                        ZStack {
                            Circle()
                                .stroke(lineWidth: 5)
                                .foregroundStyle(.gray.opacity(0.3))
                            
                                .frame(width: 40, height: 40, alignment: .center)
                            
                            Circle()
                                .trim(from: 0, to: CGFloat(1 - (timeRemaining / 60)))
                                .stroke(style: StrokeStyle(lineWidth: 5, lineCap: .round, lineJoin: .round))
                                .rotationEffect(.degrees(-90))
                                .frame(width: 40, height: 40, alignment: .center)
                                .foregroundStyle(.red)
                            
                            Text(formattedTime())
                                .font(.caption)
                                .bold()
                        }
                    }
                    .padding(.leading, 35)
                    Spacer()
                    
                    Image("logo")
                        .resizable()
                        .frame(width: 80, height: 50)
                    //.padding(.leading, 30)
                    
                    Image("Jazil")
                        .resizable()
                        .frame(width: 60, height: 60)
                        .padding(.vertical, animate ? 0 : 3)
                        .shadow(
                            color: animate ? Color.accentColor : Color("SecondaryColor"),
                            radius: animate ? 30 : 10,
                            x: 0.0,
                            y: animate ? 20 : 5
                        )
                        .scaleEffect(animate ? 1.03 : 1.0)
                        .offset(y: animate ? -5 : 0)
                    //                        .onAppear {
                    //                            animation()
                    //                        }
                }
                .onChange(of: worngAnswer) {
//                    if worngAnswer == 0 && viewModel.strick == 1 {
//                        worngAnswer += 2
//                    }
                    if worngAnswer >= 1 || viewModel.strick >= 1 {
                        stopTimer()
//                        timeRemaining = 60
                        startTimer()
                       
                        showFirstX.toggle()
                        print("Alert should show now", viewModel.strick)
                    }
                    if worngAnswer >= 2 || viewModel.strick >= 2{
                        stopTimer()
//                        timeRemaining = 60
                        startTimer()
                        showSecondX.toggle()
                        print("Alert should show now", viewModel.strick)
                    }
                    if worngAnswer >= 3 || viewModel.strick >= 3 {
                        stopTimer()
//                        timeRemaining = 60
                     
             
                        showThirdX.toggle()
                        showAlert = true
                        print("Alert should show now", viewModel.strick)
                    }
                    print(worngAnswer)
                    
                    
                }
                .alert(isPresented: $showAlert) {
                    
                 
                    Alert(
                        title: Text("Strike Alert"),
                        message: Text("لقد خسرت المساجلة الشعرية "),
                        dismissButton: .destructive(Text("Try Again")) {
                            // Delay navigation to avoid conflicts with the alert
                            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                                navigateToHome = true
                            }
                        }
                    )
                }
                .onChange(of: viewModel.strick) {
                    if worngAnswer >= 0 && viewModel.strick >= 1 {
                        worngAnswer += 1
                    
                    }
                    if  viewModel.strick >= 1 || worngAnswer >= 1  {
                        
                        stopTimer()
                        timeRemaining = 60
                        startTimer()
                       
                        showFirstX.toggle()
                        print("Alert should show now", viewModel.strick)
                    }
                    if viewModel.strick >= 2  || worngAnswer >= 2 {
                        stopTimer()
                        timeRemaining = 60
                        startTimer()
                        showSecondX.toggle()
                        print("Alert should show now", viewModel.strick)
                    }
                    if viewModel.strick >= 3  || worngAnswer >= 3 {
                        stopTimer()
//                        timeRemaining = 60
                     
             
                        showThirdX.toggle()
                        showAlert = true
                        print("Alert should show now", viewModel.strick)
                    }
                    print(worngAnswer)
                }
                //                if let session = viewModel.gameSession  {
                ScrollView {
                    ForEach(viewModel.chatMessages) { message in
                        HStack {
                            //                            message.isUser = false
                            if message.isUser {
                                Spacer()
                                Text(message.message)
                                    .padding()
                                    .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                                    .foregroundColor(.white)
                                    .cornerRadius(12)
                                
                                
                            } else {
                                
                                Text(message.message)
                                    .padding()
                                    .background(Color.gray.opacity(0.5))
                                    .cornerRadius(12)
                            }
                            Spacer()
                            
                        }
                        
                        .padding(.horizontal)
                        .padding(.vertical, 4)
                    }
                }
                
                
                
                
                
                
                
                Spacer()
                HStack {
                    TextField("Enter a message", text: $messageText)
                        .padding()
                        .background(.gray.opacity(0.1))
                        .cornerRadius(12)
                    
                    Button(action: {
                        // reset the timer when the user sends a message
                        //                        timeRemaining = 60
//                        print("verses no is : ", viewModel.isVerse)
//                        if viewModel.isVerse == true {
//                            startTimer()
//                            
//                        }
//                        print("Alert should show now", viewModel.strick)
                        // Condition that sends the message when it's long enough (Changable)
                        if messageText.count <= 4 {
                            // Do nothing if message is too short
                            let chat = ChatMessage(message: "يجيب ", isUser: false)
                            viewModel.chatMessages.append(chat)
                        } else {
                            
                            sendMessage(message: messageText)  // Pass the
              
                        }
                    }, label: {
                        Text("Send")
                            .bold()
                    })
                    
                }
            }
            
            .onAppear {
                print("Creating game session...")
                viewModel.createGameSession(difficulty: selectedDifficulty)
                
                // Observe the game session's state and start the timer when a session is created
                viewModel.$gameSession.sink { session in
                    if session != nil {
                        startTimer()  // Start the timer only when the session is created
                    }
                }.store(in: &cancellables)  // Store the subscription to handle the Combine publisher
            }
            

        
    
            
         
            
            
            .padding()
            
            .onAppear {
                isRunning = false
                if isRunning {
                    startTimer()
                }
            }
            NavigationLink(
                destination: ContentView()
                       .navigationBarBackButtonHidden(true), // Hide the back button in the destination view
                   isActive: $navigateToHome)
            {
                            EmptyView() // This link will not be shown directly
                        }
        }
    }
    
    func animation() {
        guard !animate else { return } // to make sure animate is false
        DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            withAnimation(
                Animation
                    .easeInOut(duration: 1.5)
                    .repeatForever() // to make it repeat forever
            ) {
                animate.toggle()
            }
        }
    }

    func scrollToBottom(proxy: ScrollViewProxy) {
            if let lastMessageId = viewModel.chatMessages.last?.id {
                withAnimation {
                    proxy.scrollTo(lastMessageId, anchor: .bottom)
                }
            }
        }


//    func sendMessage(message: String) {
//        // Create the user chat message with the text field value
//        let userChat = ChatMessage(message: message, isUser: true)
//        viewModel.chatMessages.append(userChat)
////        timeRemaining = 60
//        stopTimer()
//        // Submit the user move with the entered message
//        viewModel.submitMove(userMove: message)  // Pass the message to the API
//        timeRemaining = 60
//        startTimer()
//        // Clear the messageText field after sending
//        messageText = ""
//
//    
//       
//    }
    func sendMessage(message: String) {
        // Create the user chat message with the text field value
        let userChat = ChatMessage(message: message, isUser: true)
        viewModel.chatMessages.append(userChat)
        
        // Stop the timer while waiting for the response
        stopTimer()

        // Submit the user move with the entered message
        viewModel.submitMove(userMove: message)  // Pass the message to the API
        
        // Clear the messageText field after sending
        messageText = ""
        
        // Listen for AI response and restart the timer when a response is received
        viewModel.$chatMessages
            .dropFirst()  // Skip the initial state
            .sink { messages in
                if let lastMessage = messages.last, !lastMessage.isUser {
                    // AI has responded, restart the timer
                    timeRemaining = 60
                    startTimer()
                }
            }
            .store(in: &cancellables)
    }



    func formattedTime() -> String {
        let minutes = Int(timeRemaining) / 60
        let seconds = Int(timeRemaining) % 60
        return String(format: "%02d:%02d", minutes, seconds)
    }

    func startTimer() {
        isRunning = true
        timer?.invalidate()  // Invalidate any existing timer
        timer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { _ in
            if timeRemaining > 0 {
                timeRemaining -= 1
            } else {
                // User didn't answer in time, apply penalty (wrong answer)
                worngAnswer += 1
                if worngAnswer == 0 && viewModel.strick == 1 {
                    worngAnswer += 2
                }
                
                if worngAnswer == 1 || viewModel.strick == 1 { showFirstX.toggle()
                    // Automatically reset the timer
                    stopTimer()
                    timeRemaining = 60
                    startTimer()
                }
                if worngAnswer == 2 || viewModel.strick == 2 { showSecondX.toggle()
                    // Automatically reset the time
                    stopTimer()
                    timeRemaining = 60
                    startTimer()
                    }
                if worngAnswer == 3 || viewModel.strick == 3
                { showThirdX.toggle()
                    // Automatically reset the timer
                    stopTimer()
                 
                    showAlert = true
                }

     
            }
        }
        print("start timer wrong ", worngAnswer)
    }

    func stopTimer() {
        isRunning = false
        timer?.invalidate()
        timer = nil  // Reset the timer reference
    }

}

#Preview {
    ChatView(selectedDifficulty: .easy)
}



enum MessageSender {
    case me
    case allam
}

extension View {
    @ViewBuilder
    func wiggle(_ animate: Bool) -> some View {
        self
            .keyframeAnimator(initialValue: CGFloat.zero, trigger: animate) { view, value in
                view
                    .offset(x: value)
            } keyframes: { _ in
                KeyframeTrack {
                    CubicKeyframe(0, duration: 0.1)
                    CubicKeyframe(-5, duration: 0.1)
                    CubicKeyframe(5, duration: 0.1)
                    CubicKeyframe(-5, duration: 0.1)
                    CubicKeyframe(5, duration: 0.1)
                    CubicKeyframe(-5, duration: 0.1)
                    CubicKeyframe(5, duration: 0.1)
                    CubicKeyframe(0, duration: 0.1)
                }
            }
    }
}

struct ChatMessage: Identifiable {
    let id = UUID()
    let message: String
    let isUser: Bool
}

#Preview {
    ChatView(selectedDifficulty: .easy)
}
