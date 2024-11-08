//
//  D.swift
//  Jazel
//
//  Created by Muneera Y on 05/05/1446 AH.
//
enum Difficulty: String, CaseIterable, Identifiable {
    case easy = "easy"
    case hard = "hard"
    
    var id: String { self.rawValue }
}


