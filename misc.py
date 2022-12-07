#################################
#      PROJECT ASSETS:          #
#################################

import os
import sys
import numpy as np

# Directories info:
GAME_FOLDER = os.path.dirname(__file__)

# Chess constants:
WHITE = 'w'
BLACK = 'b'
EMPTY_SQR = '-'
ILLEGAL_SQR = '#'
WHITE_PIECES = ('P', 'R', 'N', 'B', 'Q', 'K')
BLACK_PIECES = ('p', 'r', 'n', 'b', 'q', 'k')
NULL_MOVE = '0000'

KINGSIDE_EMPTY = [EMPTY_SQR, EMPTY_SQR]
QUEENSIDE_EMPTY = [EMPTY_SQR, EMPTY_SQR, EMPTY_SQR]


def c_right_by_idx(i) -> str:
    if i in range(21, 25): return 'Q'
    if i in range(25, 29): return 'K'
    if i in range(91, 95): return 'q'
    if i in range(95, 99): return 'k'
    return '-'


CASTLING_PIECES = {WHITE: ('K', 'R'),
                   BLACK: ('k', 'r')}
PLAYER_RIGHTS = {WHITE: 'KQ',
                 BLACK: 'kq'}
CASTLING_RIGHTS = {WHITE: {'e1g1': 'K', 'e1c1': 'Q'},
                   BLACK: {'e8g8': 'k', 'e8c8': 'q'}}
CASTLING_EMPTY_PATH = {'K': ['K', EMPTY_SQR, EMPTY_SQR, 'R'],
                       'Q': ['R', EMPTY_SQR, EMPTY_SQR, EMPTY_SQR, 'K'],
                       'k': ['k', EMPTY_SQR, EMPTY_SQR, 'r'],
                       'q': ['r', EMPTY_SQR, EMPTY_SQR, EMPTY_SQR, 'k']}
CASTLING_PATH_IDX = {'K': (25, 29),
                     'Q': (21, 26),
                     'k': (95, 99),
                     'q': (91, 96)}
CASTLING_MOVES = {'K': 'e1g1',
                  'Q': 'e1c1',
                  'k': 'e8g8',
                  'q': 'e8c8'}

FEN_START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # white=upper case; black=lower case
FEN_TEST1 = "8/8/8/4p1K1/2k1P3/8/8/8 b - - 0 1"  # almost an entirely empty board
FEN_CAPTURING_TEST1 = "8/8/8/4p1K1/2kBPR2/8/8/8 b - - 0 1"
FEN_CAPTURING_TEST2 = "8/8/8/6K1/4kq3/8/8/8 b - - 0 1"
FEN_TRICKY_TEST = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
FEN_PROMOTION_TEST = "r3k2r/p1PpqPb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PpPpBPPP/R3K2R w KQkq - 0 1"
FEN_ENDGAME_TEST = "r7/4kp1p/p2p3b/3P4/3P3P/4nKN1/pq2PNb1/8 w - e6 29 58"
FEN_KILLER_POSITION = "rnbqkb1r/pp1p1pPp/8/2p1pP2/1P1P4/3P3P/P1P1P3/RNBQKBNR w KQkq e6 0 1"
FEN_EMPTY_POSITION = "8/8/8/8/8/8/8/8 w KQkq - 0 0"
PIECES_VALUES = {"P": 1,
                 "N": 3,
                 "B": 3,
                 "R": 5,
                 "Q": 9,
                 "K": 200,
                 "CHECK": 30,
                 "HIGHEST": sys.maxunicode}
l, h, x = -0.5, 0.5, -1 * PIECES_VALUES['HIGHEST']
PST = {'center': np.array([x, x, x, x, x, x, x, x, x, x,
                           x, x, x, x, x, x, x, x, x, x,
                           x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                           x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                           x, 0, 0, l, l, l, l, 0, 0, x,
                           x, 0, 0, l, h, h, l, 0, 0, x,
                           x, 0, 0, l, h, h, l, 0, 0, x,
                           x, 0, 0, l, l, l, l, 0, 0, x,
                           x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                           x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                           x, x, x, x, x, x, x, x, x, x,
                           x, x, x, x, x, x, x, x, x, x]),
       'K': np.array([x, x, x, x, x, x, x, x, x, x,
                      x, x, x, x, x, x, x, x, x, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, h, 0, h, 0, h, 0, x,
                      x, x, x, x, x, x, x, x, x, x,
                      x, x, x, x, x, x, x, x, x, x], ),
       'k': np.array([x, x, x, x, x, x, x, x, x, x,
                      x, x, x, x, x, x, x, x, x, x,
                      x, 0, 0, h, 0, h, 0, h, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, 0, 0, 0, 0, 0, 0, 0, 0, x,
                      x, x, x, x, x, x, x, x, x, x,
                      x, x, x, x, x, x, x, x, x, x], )
       }
ARRAY_B_DIM = 8

# Padded array maps:
PADDED_B_WIDTH = 10
PADDED_B_HEIGHT = 12
PADDED_CORNERS_IDX = (21, 28, 91, 98)
PADDED_WALLS_IDX = [110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                    20, 30, 40, 50, 60, 70, 80, 90,
                    29, 39, 49, 59, 69, 79, 89, 99,
                    10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
PADDED_B_EMPTY = ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                  '#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '-', '-', '-', '-', '-', '-', '-', '-', '#',
                  '#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                  '#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
PADDED_B_NOTATIONS = ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                      '#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                      '#', 'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', '#',
                      '#', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', '#',
                      '#', 'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', '#',
                      '#', 'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4', '#',
                      '#', 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5', '#',
                      '#', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6', '#',
                      '#', 'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', '#',
                      '#', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8', '#',
                      '#', '#', '#', '#', '#', '#', '#', '#', '#', '#',
                      '#', '#', '#', '#', '#', '#', '#', '#', '#', '#']
PADDED_NOTT_TO_IDX = {'a8': 91, 'b8': 92, 'c8': 93, 'd8': 94, 'e8': 95,
                      'f8': 96, 'g8': 97, 'h8': 98,
                      'a7': 81, 'b7': 82, 'c7': 83, 'd7': 84, 'e7': 85,
                      'f7': 86, 'g7': 87, 'h7': 88,
                      'a6': 71, 'b6': 72, 'c6': 73, 'd6': 74, 'e6': 75,
                      'f6': 76, 'g6': 77, 'h6': 78,
                      'a5': 61, 'b5': 62, 'c5': 63, 'd5': 64, 'e5': 65,
                      'f5': 66, 'g5': 67, 'h5': 68,
                      'a4': 51, 'b4': 52, 'c4': 53, 'd4': 54, 'e4': 55,
                      'f4': 56, 'g4': 57, 'h4': 58,
                      'a3': 41, 'b3': 42, 'c3': 43, 'd3': 44, 'e3': 45,
                      'f3': 46, 'g3': 47, 'h3': 48,
                      'a2': 31, 'b2': 32, 'c2': 33, 'd2': 34, 'e2': 35,
                      'f2': 36, 'g2': 37, 'h2': 38,
                      'a1': 21, 'b1': 22, 'c1': 23, 'd1': 24, 'e1': 25,
                      'f1': 26, 'g1': 27, 'h1': 28}

# Pieces table maps:
PIECES_IDX_TABLE_EMPTY = {'b': {'k': [],
                                'q': [],
                                'r': [],
                                'n': [],
                                'b': [],
                                'p': []},
                          'w': {'K': [],
                                'Q': [],
                                'R': [],
                                'N': [],
                                'B': [],
                                'P': []}}
PIECES_IDX_TABLE_START = {'b': {'k': [95],
                                'q': [94],
                                'r': [91, 98],
                                'n': [92, 97],
                                'b': [93, 96],
                                'p': [81, 82, 83, 84, 85, 86, 87, 88]},
                          'w': {'K': [25],
                                'Q': [24],
                                'R': [21, 28],
                                'N': [22, 27],
                                'B': [23, 26],
                                'P': [31, 32, 33, 34, 35, 36, 37, 38]}}
PIECES_HASH_TABLE_START = {'b': {'r': ['a8', 'h8'],
                                 'n': ['b8', 'g8'],
                                 'b': ['c8', 'f8'],
                                 'q': ['d8'],
                                 'k': ['e8'],
                                 'p': ['a7', 'b7', 'c7', 'd7', 'e7', 'f7',
                                       'g7', 'h7']},
                           'w': {
                               'P': ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2',
                                     'h2'],
                               'Q': ['d1'],
                               'K': ['e1'],
                               'B': ['c1', 'f1'],
                               'N': ['b1', 'g1'],
                               'R': ['a1', 'h1']}}

# Note: inspired by the directions mapping from Sunfish open-source chess engine
# Directions written in white-player perspective
F, B, R, L = PADDED_B_WIDTH, -1 * PADDED_B_WIDTH, 1, -1
REG_MASKS = {
    # A dictionary for moves by piece symbol. Moves are checked by written order.
    "P": F,
    "R": (F, B, R, L),
    "N": (F + F + R, F + F + L, R + R + F, R + R + B, L + L + F, L + L + B,
          B + B + R, B + B + L),
    "B": (F + R, F + L, B + R, B + L),
    "Q": (F, R, L, B, F + R, F + L, B + R, B + L),
    "K": (F, F + R, F + L, R, L, B, B + R, B + L),
    "p": B,
    "r": (B, F, R, L),
    "n": (B + B + R, B + B + L, R + R + B, R + R + F, L + L + B, L + L + F,
          F + F + R, F + F + L),
    "b": (B + R, B + L, F + R, F + L),
    "q": (B, R, L, F, B + R, B + L, F + R, F + L),
    "k": (B, B + R, B + L, R, L, F, F + R, F + L)}

SPECIAL_MASKS = {
    "P": (F + R, F + L),
    "p": (B + R, B + L),
    "K": (R + R, L + L),
    "k": (R + R, L + L)}
