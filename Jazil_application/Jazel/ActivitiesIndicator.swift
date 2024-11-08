import SwiftUI

struct ActivitiesIndicator: View {
    @State private var shouldAnimate = false
    
    var body: some View {
        
             
                 Section {
                     HStack {
                         Circle()
                             .fill(Color.accentColor)
                             .frame(width: 20, height: 20)
                             .scaleEffect(shouldAnimate ? 1.0 : 0.5)
                             .animation(Animation.easeInOut(duration: 0.5).repeatForever())
                         Circle()
                             .fill(Color("SecondaryColor"))
                             .frame(width: 20, height: 20)
                             .scaleEffect(shouldAnimate ? 1.0 : 0.5)
                             .animation(Animation.easeInOut(duration: 0.5).repeatForever().delay(0.3))
                         Circle()
                             .fill(Color.accentColor)
                             .frame(width: 20, height: 20)
                             .scaleEffect(shouldAnimate ? 1.0 : 0.5)
                             .animation(Animation.easeInOut(duration: 0.5).repeatForever().delay(0.6))
                     }
                 }
             .onAppear {
                 self.shouldAnimate = true
             }
         }
         
     }


#Preview {
    ActivitiesIndicator()
}
