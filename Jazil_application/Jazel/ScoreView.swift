//  ScoreView.swift
//  Jazl
//
//  Created by Mohanned Alsahaf on 05/05/1446 AH.
//

import SwiftUI

struct ScoreView: View {
    
    
    
    
    @State var friendList:[friend] = [friend(id: UUID().uuidString, counter: 1 ,name:"مهند",score:41000),
                                      friend(id: UUID().uuidString, counter: 2 ,name: "متعب القني", score: 31213)
                                      ,friend(id: UUID().uuidString, counter: 3 ,name: "محمد التنم", score: 22213),
                                      friend(id: UUID().uuidString, counter: 4 ,name: "محمد العمران", score: 21213),
                                      friend(id: UUID().uuidString, counter: 5 ,name: "محمد الحرز", score: 20216)
                                      ,friend(id: UUID().uuidString, counter: 6 ,name:"مهند",score:11912),
                                      friend(id: UUID().uuidString, counter: 7 ,name: "متعب القني", score: 12345)
                                      ,friend(id: UUID().uuidString, counter: 8 ,name: "حمد ", score: 31213),
                                      friend(id: UUID().uuidString, counter: 9 ,name: "ياسين الصحاف", score: 12)
                                      
                                      
                                      
    
    ]
    
    var body: some View {
        ZStack{
            LinearGradient(colors: [Color.accentColor.opacity(0.5),Color.white], startPoint: .top, endPoint: .bottom).ignoresSafeArea()
            
            VStack{
              
                Spacer()
                ZStack{
                    
                    Image("banner")
                        .resizable()
                        .frame(width: 400, height: 400)
                        
                     
                    Text("نوابغ الشعر")
                        .font(.largeTitle)
                        .bold()
                        .foregroundStyle(Color(red: 98/255, green: 113/255, blue: 164/255))
                        .padding(.bottom,50)
                   
                     
                }
              
                
                HStack{
                    Spacer()
                    ZStack{
                        Rectangle()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .foregroundStyle(Color(red: 98/255, green: 113/255, blue: 164/255))
                            .cornerRadius(12)
                            .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                            .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
                            .frame(width: 170, height: 105, alignment: .leading)
                            .offset(y:-100)
                        VStack{
                            Image("global")
                                .resizable()
                                .frame(width: 50, height: 50)
                                .offset(y:-110)
                                .padding(.top, 15)
                            Text("عالمياً")
                                .offset(y:-115)
                                .bold()
                                .font(.title2)
                                .foregroundStyle(Color.accentColor)
                        }
                        .padding(.top, /*@START_MENU_TOKEN@*/10/*@END_MENU_TOKEN@*/)
                    }
                    
                    
                    Spacer()
                    ZStack{
                        Rectangle()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                            .foregroundStyle(Color(red: 98/255, green: 113/255, blue: 164/255))
                            .cornerRadius(12)
                            .shadow(color: Color.black.opacity(0.3), radius: 10, x: 10, y: 10) // Bottom right shadow
                            .shadow(color: Color.white.opacity(0.6), radius: 10, x: -5, y: -5) // Top left "highlight"
                            .frame(width: 170, height: 105, alignment: .leading)
                            .offset(y:-100)
                        VStack{
                            Image("friend")
                                .resizable()
                                .frame(width: 50, height: 50)
                                .offset(y:-110)
                                .padding(.top, 15)
                            Text("الأصدقاء")
                                .offset(y:-115)
                                .bold()
                                .font(.title2)
                                .foregroundStyle(Color.accentColor)
                        }
                        .padding(.top, /*@START_MENU_TOKEN@*/10/*@END_MENU_TOKEN@*/)
                    }
                    Spacer()
                        
                }
            
                
                
      
                    List {
                        Section{
                            ForEach(friendList,id: \.id){friend in
                                HStack{
                                    Text(String(friend.score))
                                    Image("coin")
                                        .resizable()
                                        .frame(width: 25, height: 25)
                                    Spacer()
                                    
                                    Text(friend.name)
                                        .font(.title)
                                    Image("person")
                                        .resizable()
                                        .frame(width: 35, height: 35)
                                    Text(".")
                                    Text(String(friend.counter))
                                        .font(.title3)
                                    
                                    
                                   
                                }
                                
                                
                                
                                
                                
                            }
                            .listRowBackground(LinearGradient(colors: [Color.accentColor,Color.accentColor.opacity(0.1)], startPoint: .top, endPoint: .top))
                            
                           
                        }header: {
                            
                            Text("النوابغ")
                                .font(.title3)
                                .bold()
                                .foregroundStyle(Color.accentColor)
                                .padding(.leading,260)
                            
                            
                        }
                    }.offset(y:-90)
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)
                    .frame(width: 400, height: 500, alignment: .center)
                 
                    
                   
            
                   
                    
                  
                
                
                
                
            }
          
            
            
        }
    }
}

#Preview {
    ScoreView()
}


struct friend:Identifiable{
   
    
    let id:String
    let counter:Int
    let name:String
    let score:Int
}
