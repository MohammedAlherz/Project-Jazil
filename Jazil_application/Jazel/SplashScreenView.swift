//
//  SplashScreenView.swift
//  gameday
//
//  Created by Muneera Y on 04/05/1446 AH.
//

import SwiftUI

struct SplashScreenView: View {
    @State private var isActive = false
    @State private var scaleEffect = 0.6
    @State private var opacity = 0.0
    
    var body: some View {
        if isActive {
            ContentView()
        } else {
            ZStack{
                Image("bg")
                    .resizable()
                    .opacity(0.05)
                LinearGradient(
                    gradient: Gradient(stops: [
                        .init(color: Color.white.opacity(0.3), location: 0),
                        .init(color: Color(red: 1.0, green: 166/255, blue: 59/255).opacity(0.5), location: 0.42)
                    ]),
                    startPoint: .top,
                    endPoint: .bottom
                ).edgesIgnoringSafeArea(.all)
                VStack {
                    Image("logo")
                        .resizable()
                        .scaledToFit()
                        .frame(width: 300, height: 150)
                        .foregroundColor(.blue)
                        .scaleEffect(scaleEffect)
                        .opacity(opacity)
                        .onAppear {
                            withAnimation(.easeIn(duration: 1.0)) {
                                scaleEffect = 1.0
                                opacity = 1.0
                            }
                        }
                    
                }}
            .onAppear {
                DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                    withAnimation {
                        isActive = true
                    }
                }
            }
        }
    }
}
#Preview {
    SplashScreenView()
}
