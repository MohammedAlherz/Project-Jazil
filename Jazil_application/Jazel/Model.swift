//
//  Model.swift
//  gameday
//
//  Created by Muneera Y on 27/04/1446 AH.
//
struct GameSession: Codable {
    let session_id: String
    let message: String
    let required_verses: Int
    let current_verses: Int
    let strikes: Int
    let last_letter: String?
    let ai_response: String?
    let game_over: Bool
    let game_result: String
}
