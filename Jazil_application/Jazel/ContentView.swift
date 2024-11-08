//
//  home.swift
//  gameday
//
//  Created by Muneera Y on 01/05/1446 AH.
//

import SwiftUI

struct ContentView: View {
    let screenWidth = UIScreen.main.bounds.width
    let screenHeight = UIScreen.main.bounds.height
    
    var body: some View {
        NavigationStack {
        
         
            ZStack {
                Image("bg")
                    .resizable().opacity(0.1)
//                    .frame(width: 280, height: 210)
                VStack{
                  
                    
                    HStack {
                        Image("logo")
                            .resizable()
                            .frame(width: 200, height: 100)
                            .offset(x:100 , y: 0 )
                        
                        ZStack{
                            Rectangle()
                                .frame(width: 100, height: 30)
                                .foregroundStyle( LinearGradient(colors: [Color("SecondaryColor").opacity(0.5),Color(red: 98/255, green: 113/255, blue: 164/255)], startPoint: .bottom, endPoint: .top))
                                .cornerRadius(12)
                                .padding(.trailing,10)
                            
                            HStack{
                                Text("1280")
                                    .font(.title3)
                                    .bold()
                                    .italic()
                                    .foregroundStyle(Color.accentColor)
                                    .padding(.leading, 13)
                                
                                
                                Image("coin")
                                    .resizable()
                                    .frame(width: 50, height: 50)
                                    .bold()
                                
                                
                                
                            }.padding(.leading, 20)
                        }.padding()
                        .offset(x:25 , y: -40 )
                        
                    }
                    
                    
//                    Spacer()
                    
                    
                    
                    // Game Start Button
                    NavigationLink(destination: DifficultySelectionView()) {
                        VStack {
                            Image("IMG_5600")
                                .resizable()
                                .frame(width: 110, height: 110)
                            
                            Text("العب الان")
                                .font(.title2)
                                .bold()
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
//                        .background(Color(red: 153/255, green: 177/255, blue: 255/255))
                        .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                        .cornerRadius(12)
                        .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                        .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
                        
                    }
                    .foregroundColor(.white)
                    .frame(width: 266, height: 165, alignment: .leading)
                    .padding()
                    //                        .shadow(color: Color.black.opacity(0.6), radius: 10, x: 0, y: 5)
                    // Instructions Button
                    NavigationLink(destination: GameExplainView()) {
                        VStack {
                            Image("Image") // Replace with your image
                                .resizable()
                                .frame(width: 110, height: 110)
                            Text("التعليمات")
                                .font(.title2)
                                .bold()
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                        .cornerRadius(12)
                        .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                        .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
                    }
                    .foregroundColor(.white)
                    .frame(width: 266, height: 165, alignment: .leading)
                    
                    // Points Button
                    NavigationLink(destination: ScoreView()) {
                        VStack {
                            Image("crown")  // Replace with your image
                                .resizable()
                                .frame(width: 112, height: 112)
                            Text("نوابغ الشعر")
                                .font(.title2)
                                .bold()
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .background(Color(red: 98/255, green: 113/255, blue: 164/255))
                        .cornerRadius(12)
                        .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                        .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5)
                    }
                    .foregroundColor(.white)
                    .frame(width: 266, height: 165, alignment: .leading)
                    .padding()
                    
//                    HStack {
//                        // Settings icon
//
//
//
//
//
//
//                        // Image on the right
//                        Image("Image 2")
//                            .resizable()
//                            .frame(width: 70, height: 70)
//                            .clipShape(Circle()) // Optional: if you want it circular
//                            .padding(.trailing, 16)
//                            .offset(x: 15)
//                        Spacer()
//                    }
                    
                }
            }.edgesIgnoringSafeArea(.top)
                .background(LinearGradient(
                    gradient: Gradient(stops: [
                        .init(color: Color.white.opacity(0.3), location: 0), // 8% opacity at 0% position
                        .init(color: Color(red: 1.0, green: 166/255, blue: 59/255).opacity(0.5), location: 0.42) // FFA63B at 42% position
                        /*    .init(color: Color(red: 153/255, green: 177/255, blue: 255/255).opacity(0.1), location: 1.0)  */ // Light blue at 100% position
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                ))
                .foregroundColor(.black)
            
            
        }
            
        }
//    }
}



struct InstructionsView: View {
    var body: some View {
        VStack {
            Text("Instructions go here")
                .font(.largeTitle)
                .padding()
            // Add content for the Instructions page
        }
    }
}

struct PointsView: View {
    var body: some View {
        VStack {
            Text("Your Points")
                .font(.largeTitle)
                .padding()
            // Add content for Points page
        }
    }
}

#Preview {
    ContentView()
}
