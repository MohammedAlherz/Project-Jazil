//
//  GameExplainView.swift
//  gameday
//
//  Created by Muneera Y on 01/05/1446 AH.
//

import SwiftUI

struct GameExplainView: View {
    @State var animate:Bool = false
    @State var textss: String = """
    لعبة جازل: تحدي الشعر العربي

    طريقة اللعب:

    اختر مستوى:
    سهل: اكتب 6 أبيات.
    صعب: اكتب 10 أبيات.

    شروط الأبيات:

    كل بيت يجب أن يكون صحيحاً نحوياً وعروضياً.
    يتكون من صدر وعجز ويتبع أحد البحور الشعرية.
    استخدم نفس حرف الروي في كل بيت.
    حرف الروي:

    هو الحرف الأخير الذي تبنى عليه القافية (قبل هاء الضمير، واو الجماعة، ألف الإطلاق، ياء المتكلم، نون التوكيد).

    نظام النقاط:

    3 محاولات خاطئة تعني الخسارة.
    المحاولات الخاطئة تشمل: خطأ نحوي/عروضي، تكرار بيت، خطأ في حرف الروي، أو تجاوز الوقت (60 ثانية).

    الفوز:

    أكمل الأبيات المطلوبة للفوز، أو تخسر إذا استنفدت محاولاتك.

    """
    var body: some View {
        
        
        ZStack{
            LinearGradient(colors: [Color.accentColor.opacity(0.5),Color.white], startPoint: .top, endPoint: .bottom).ignoresSafeArea()
            VStack{
                Image("Jazil")
                   .resizable()
                   .scaledToFit()
                    .frame(width: 250, height: 250)
                    .padding(.horizontal,animate ? 0 : 10)
                    .shadow(
                        color: animate ? Color.accentColor : Color("SecondaryColor"),
                        radius:animate ? 30:10,
                        x: 0.0,
                        y: animate ? 20:5
                            )
                    
                    .scaleEffect(animate ? 1.03 : 1.0)
                    
                    .offset(y: animate ? -5 : 0)
                    .onAppear{
                        animation()
                    }
                Spacer()
                ScrollView {
                    VStack {
                        Spacer()
                        Text(textss)
                            .font(.largeTitle)
                            .multilineTextAlignment(.trailing) // Centers text horizontally
                        Spacer()
                    }
                    .frame(maxWidth: .infinity)
                }.padding(8)
                Spacer()
                Spacer()
            }
        }
        
    }
    
    func animation(){
        guard !animate else {return} //to make sure animate is false
        DispatchQueue.main.asyncAfter(deadline: .now() + 1){
            withAnimation(
            
            Animation
                .easeInOut(duration: 1.5)
                .repeatForever()// to make it reapet forever
            
            ) {
                animate.toggle()
            }
        }
    }
}

#Preview {
    GameExplainView()
}
