//
////
////  DifficultySelectionView.swift
////  gameday
////
////  Created by Muneera Y on 01/05/1446 AH.
////
//import SwiftUI
//
//struct DifficultySelectionView: View {
//    @State private var selectedDifficulty: Difficulty = .easy
//    @StateObject var viewModel = GameViewModel()
////    var selectedDifficulty: Difficulty
//    var body: some View {
//        NavigationStack {
//            ZStack {
//                Image("bg")
//                    .resizable().opacity(0.05)
//
//
//                VStack {
//                    Spacer()
//                            Image("logo")
//                                .resizable()
//                                .frame(width: 280, height: 200)
//                                .offset(y: -100)
////                            Image("Image 2")
////                                .resizable()
////                                .frame(width: 100, height: 100)
////                                .clipShape(Circle())
////                                .padding(.trailing, 16)
////                                .offset(x: 15, y: 10.87)
//////
//
//
//                            // Difficulty Picker
////                            Text("اختر مستوى الصعوبة")
////                                .font(.title)
////
////                                .bold()
////                                .offset( y: 20)
//
//                            //                        Picker("Difficulty", selection: $selectedDifficulty) {
//                            //                            ForEach(Difficulty.allCases, id: \.self) { difficulty in
//                            //                                Text(difficulty.rawValue.capitalized).tag(difficulty)
//                            //                            }
//                            //                        }
//                            //                        .pickerStyle(SegmentedPickerStyle())
//                            //                        .padding()
//                            //                        .offset( y: 118.87)
//
//                            // NavigationLink to GameView, passing the selected difficulty
//                            NavigationLink(destination: GameView(selectedDifficulty: .easy)) {
//                                VStack {
//                                    //                                Image("IMG_5600")
//                                    //                                    .resizable()
//                                    //                                    .frame(width: 110, height: 110)
//                                    Text("المستوى السهل")
//                                        .font(.title3)
//                                        .bold()
//                                    viewModel.createGameSession(difficulty: .easy)
//                                }
//                                .frame(maxWidth: .infinity, maxHeight: .infinity)
//                                .background(Color(red: 153/255, green: 177/255, blue: 255/255))
//                                .cornerRadius(12)
//                                .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
//                                .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
//
//                            }
//                            .foregroundColor(.black)
//                            .frame(width: 266, height: 115, alignment: .leading)
//
//                            .offset( y: 30.87)
//                            NavigationLink(destination: GameView(selectedDifficulty: .hard)) {
//                                VStack {
//                                    //                                Image("IMG_5600")
//                                    //                                    .resizable()
//                                    //                                    .frame(width: 110, height: 110)
//                                    Text("المستوى الصعب")
//                                        .font(.title3)
//                                        .bold()
//                                }
//                                .frame(maxWidth: .infinity, maxHeight: .infinity)
//                                .background(Color(red: 153/255, green: 177/255, blue: 255/255))
//                                .cornerRadius(12)
//                                .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
//                                .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
//
//                            }
//                            .foregroundColor(.black)
//                            .frame(width: 266, height: 115, alignment: .leading)
//                            .offset( y: 40.87)
//
//                            Spacer()
//
//                    Image("Image 2")
//                        .resizable()
//                        .frame(width: 100, height: 100)
//                        .clipShape(Circle())
//                        .padding(.trailing, 16)
//                        .offset(x: -120 )
////
//
//                    }
//
//
//
//            } .edgesIgnoringSafeArea(.top)
//                .background(LinearGradient(
//                    gradient: Gradient(stops: [
//                        .init(color: Color.white.opacity(0.3), location: 0), // 8% opacity at 0% position
//                        .init(color: Color(red: 1.0, green: 166/255, blue: 59/255).opacity(0.5), location: 0.42) // FFA63B at 42% position
//                        /*    .init(color: Color(red: 153/255, green: 177/255, blue: 255/255).opacity(0.1), location: 1.0)  */ // Light blue at 100% position
//                    ]),
//                    startPoint: .top,
//                    endPoint: .bottom
//                ))
//                .foregroundColor(.black)
//        }
//    }
//}
//
//#Preview {
//    DifficultySelectionView()
//}
//
import SwiftUI

struct DifficultySelectionView: View {
    @State private var animate: Bool = false
    @State private var selectedDifficulty: Difficulty = .easy
    @StateObject var viewModel = GameViewModel()

    var body: some View {
        NavigationStack {
           
            ZStack {
                Image("bg")
                    .resizable()
                    .opacity(0.05)
                
                VStack {
                  
                    
                    Image("logo")
                        .resizable()
                        .frame(width: 280, height: 200)
                        .offset(y: 50)
                    
                    // Title
                    Text("اختر مستوى اللعبة")
                        .font(.title)
                        .bold()
                        .offset(y: 20)

                    // Buttons for difficulty selection
                    VStack(spacing: 20) {
                        // Easy Difficulty Button
                        NavigationLink(destination: ChatView(selectedDifficulty: selectedDifficulty)) {
                            Text("سهل")
                                .font(.largeTitle)
                                .bold()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                                .cornerRadius(12)
                                .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10)
                                .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5)
                                .padding(.horizontal)
                        }
                        .frame(width: 300, height: 165, alignment: .leading)
                    
                        .foregroundColor(.white)
                        .onTapGesture {
                            selectedDifficulty = .easy // or .hard for the other button
                            viewModel.createGameSession(difficulty: selectedDifficulty)
                        }
                        
                        // Hard Difficulty Button
                        NavigationLink(destination: ChatView(selectedDifficulty: .hard)) {
                            Text("صعب")
                                .font(.largeTitle)
                                .bold()
                                .frame(maxWidth: .infinity, maxHeight: .infinity)
                                .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                                .cornerRadius(12)
                                .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                                .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left highlight
                                .padding(.horizontal)
                        }
                        .frame(width: 300, height: 165, alignment: .leading)
                        .foregroundColor(.white)
                        .onTapGesture {
                            selectedDifficulty = .hard // or .hard for the other button
                            viewModel.createGameSession(difficulty: selectedDifficulty)
                        }
                    }
                    .padding(.top, 40)
                    
                    Spacer()
             
                        // Image at the bottom
                        Image("Image 2")
                            .resizable()
                            .frame(width: 100, height: 100)
                            .clipShape(Circle())
                            .padding(.trailing, 16)
                            .offset(x: -120)
                            .shadow(
                                color: animate ? Color.accentColor : Color("SecondaryColor"),
                                radius: animate ? 30 : 10,
                                x: 0.0,
                                y: animate ? 20 : 5
                            )
                            .scaleEffect(animate ? 1.03 : 1.0)
                            .offset(y: animate ? -5 : 0)
                     .onAppear {
                        animation()
                    }
                }
            }
            .edgesIgnoringSafeArea(.top)
            .background(LinearGradient(
                gradient: Gradient(stops: [
                    .init(color: Color.white.opacity(0.3), location: 0),
                    .init(color: Color(red: 1.0, green: 166/255, blue: 59/255).opacity(0.5), location: 0.42)
                ]),
                startPoint: .top,
                endPoint: .bottom
            ))
            .foregroundColor(.black)
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

}

#Preview {
    DifficultySelectionView()
}
