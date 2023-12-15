import chess
import chess.pgn
import io

# Your PGN string
#pgn_string = """
#1. d3 c5 2. e4 f6 3. c4 d5 4. Nc3 d4 5. Nce2 Nc6 6. Nf3 e5 7. Ng3 Bg4 8. Be2 Qc7 9. h3 Bh5 10. Nh2 Bxe2 11. Qxe2 O-O-O 12. Nf3 Kb8 13. O-O Bd6 14. a3 Nge7 15. b4 cxb4 16. axb4 Nxb4 17. Ba3 Nbc6 18. Bxd6 Qxd6 19. Qa2 a6 20. Nf5 Nxf5 21. exf5 Nb4 22. Qb3 Qc5 23. Rfb1 a5 24. Ra4 b6 25. Nd2 Kc7 26. Ne4 Qe7 27. c5 bxc5 28. Rxa5 Rb8 29. Rxc5+ Kd7 30. Qd5+ Nxd5 31. Rxd5+ Qd6 32. Rxd6+ Ke7 33. Rxb8 Rxb8 34. Rc6 Rb1+ 35. Kh2 Kf8 36. Nc5 h6 37. Rc8+ Ke7 38. Rc7+ Kf8 39. Ne6+ Kg8 40. Rxg7+ Kh8 41. Rg6 h5 42. Rxf6 Kg8 43. Rg6+ Kh7 44. Rh6+ Kxh6 45. Nf4 Kg5 46. Nd5 Rb2 47. f4+ exf4 48. f6 h4 49. f7 Rb8 50. f8=Q Rxf8 51. Ne7 Kf6 52. Nc6 1-0
#"""
#
#pgn_io = io.StringIO(pgn_string)
#
## Load the PGN string
#game = chess.pgn.read_game(pgn_io)
#game_board = game.board()
#for move in game.mainline_moves():
#    print(str(move), type(move))

with open('lichess_db_standard_rated_2014-07.pgn') as f:
    game = chess.pgn.read_game(f)
    print('cek', game.headers.get('Result'))
print(game)