"""
Author: Eran Avraham
A final CS project.

This ChessEngine module is the framework that holds and runs all the essential components together,
i.e. the Board, the moveGenerator as well as Search class and Position Evaluator.
This class runs an infinite loop that receives board position as input, and return best move
found as output; This is the main point of chess engines.

* The chess engine and the gui should be entirely separated.
* The chess engine operates simple interface through standard I/O,
    That way it can interact with users or other programs through the UCI protocol.
"""
import time as timer
from multiprocessing.pool import ThreadPool
from threading import Thread
import copy
from abc import ABC, abstractmethod
from sortedcontainers import SortedDict
from misc import *


##################################################################
#   CONSTANTS:
##################################################################

AUTHOR_NAME = "ERAN AVRAHAM"
ENGINE_NAME = "SkyFioliEngine"
PROTOCOLS = ('uci', 'dev')

##################################################################
#   BOARD:
##################################################################
"""
This is the data structure that represents a board state (pieces position) in the memory, and keeps all the data
provided by an FEN string.
* Notice: Searcher creates an instance of this data structure for every single position it is looking,
        MoveGenerator use it for calculating possible moves and Evaluator will use it to evaluate position value. 
Main approaches for board rep':
1. Square centric- a data structure that holds the information about the board, by squares.
    Best used for scanning the board and check if square is occupied. (arrays, matrices, etc..)
2. Piece centric- a data structure that holds information about pieces positions on the board.
    Could be used for quick access to a certain piece location. (bitboards, hash-tables)
"""


# class BoardState:
#     def __init__(self, array, pieces, color, castling, en_passant, hm, fm):
#         # FEN data-
#         self.array = array
#         self.pieces = pieces
#         self.color = color
#         self.castling = castling
#         self.en_passant = en_passant
#         self.hm = hm
#         self.fm = fm
#         # additional data-
#         self.enemy = BLACK if color == WHITE else WHITE
#         self.kings = {WHITE: self.pieces['K'][0],
#                       BLACK: self.pieces['k'][0]}


class Board(ABC):
    def __init__(self):
        self.array = []
        self.pieces = {}
        self.color, self.castling, self.en_passant, self.hm, self.fm = [''] * 5
        self.enemy = ''
        #self.history_log = []

    @abstractmethod
    def parse_fen(self, fen_str):
        """ parse_fen() receives an FEN string, parse it, and return a tuple with the parsed data."""
        pass

    @abstractmethod
    def get_fen(self) -> str:
        """ encode board into FEN string."""
        pass

    @abstractmethod
    def make_move(self, move):
        """ This method receives move to execute,
            and returns a new Board instance after updating positions AND data."""
        pass


class PaddedArrayBoard(Board):
    """ The PaddedArrayBoard uses a padded-1D-array, in the total size of 12*10.
        Empty/Illegal squares have their unique marks, and squares indexing is according
        to: a1=21, h1=28, a8=91, h8=98."""

    def __init__(self):
        Board.__init__(self)
        self.parse_fen(FEN_START)
        self.generator = MoveGenerator()

    def parse_fen(self, fen_str):
        """Reset the board according to a FEN string"""
        self.array = copy.deepcopy(PADDED_B_EMPTY)  # a 12*10 sized 1D array
        self.pieces = copy.deepcopy(PIECES_IDX_TABLE_EMPTY)
        placement_str, self.color, self.castling, self.en_passant, self.hm, self.fm = fen_str.split()
        self.hm, self.fm = int(self.hm), int(self.fm)
        self.enemy = BLACK if self.color == WHITE else WHITE
        i = PADDED_NOTT_TO_IDX['a8']  # getting first index of 8'th rank (=91)
        for c in placement_str:
            if str(c).isnumeric():  # numbers
                i += int(c)
            elif str(c).isalpha():  # pieces
                self.array[i] = c
                p_clr = WHITE if str(c).isupper() else BLACK
                self.pieces[p_clr][c].append(i)
                i += 1
            elif c == '/':  # end of rank
                i -= ((PADDED_B_WIDTH * 2) - 2)
            else:
                print("fen string error")
                return

    def get_fen(self) -> str:
        fen = ''
        # encode array-
        for i in range(ARRAY_B_DIM):
            rank = self.array[91 - i * 10:99 - i * 10]
            ecc = 0     # empty cells count
            for c in rank:
                if c.isalpha():
                    if ecc != 0:
                        fen += str(ecc)
                        ecc = 0
                    fen += c
                elif c == EMPTY_SQR:
                    ecc += 1
            if ecc != 0:
                fen += str(ecc)
                ecc = 0
            fen += '/'
        # encode params-
        fen = fen[0:-1]
        fen += f' {self.color} {self.castling} {self.en_passant} {self.fm} {self.hm}'
        return fen

    def make_move(self, move):
        bordy = copy.deepcopy(self)
        mv_src, mv_dest = move[:2], move[2:4]
        src_idx, dest_idx = PADDED_NOTT_TO_IDX[mv_src], PADDED_NOTT_TO_IDX[mv_dest]
        moving_piece, target_sqr = bordy.array[src_idx], bordy.array[dest_idx]
        # TODO Can comprehend with move types: reg, capturing, castling, en-passant, promotion:
        # Promotion move:
        if len(move) == 4:
            # castling move:
            if (moving_piece in 'Kk'
                    and move in CASTLING_MOVES.values()) \
                    and (c_right_by_idx(dest_idx) in bordy.castling):
                bordy.make_castling(dest_idx=dest_idx)
            # regular or capturing move:
            elif bordy.is_sqr_empty(dest_idx) or bordy.is_sqr_enemy(dest_idx, perspective=bordy.color):
                (bordy.pieces[bordy.color][moving_piece]).remove(src_idx)
                if target_sqr != EMPTY_SQR:     # capturing verified
                    (bordy.pieces[bordy.enemy][target_sqr]).remove(dest_idx)
                (bordy.pieces[bordy.color][moving_piece]).append(dest_idx)
                bordy.array[dest_idx], bordy.array[src_idx] = bordy.array[src_idx], EMPTY_SQR
        elif len(move) == 5 and moving_piece in ('P', 'p'):
            promote = move[-1]
            bordy.pieces[bordy.color][moving_piece].remove(src_idx)
            bordy.pieces[bordy.color][promote].append(dest_idx)
            if target_sqr.isalpha():   # if we capture a rival piece
                bordy.pieces[bordy.enemy][target_sqr].remove(dest_idx)
            bordy.array[dest_idx], bordy.array[src_idx] = promote, EMPTY_SQR

        # Update castling rights:
        if moving_piece in 'Kk' and src_idx in (25, 95):    # player king first move
            for r in PLAYER_RIGHTS[bordy.color]:
                bordy.remove_castling_right(r)
        if target_sqr in 'Kk':  # enemy king captured
            for r in PLAYER_RIGHTS[bordy.enemy]:
                bordy.remove_castling_right(r)
        if moving_piece in 'Rr' and src_idx in PADDED_CORNERS_IDX:  # player rook first move
            bordy.remove_castling_right(c_right_by_idx(src_idx))
        if target_sqr in 'Rr' and dest_idx in PADDED_CORNERS_IDX:   # enemy rook captured at origin
            bordy.remove_castling_right(c_right_by_idx(dest_idx))

        # New board final setup:
        bordy.hm = self.hm + 1  # increment half-moves counter
        bordy.fm = bordy.hm // 2    # update full-moves counter
        bordy.color, bordy.enemy = bordy.enemy, bordy.color  # change color (pass turn to other player)
        return bordy

    def make_castling(self, dest_idx):
        k_symb, r_symb = CASTLING_PIECES[self.color]
        if dest_idx == 27:  # white to kingside
            (self.pieces[self.color][k_symb]) = [27]
            (self.pieces[self.color][r_symb]).remove(28)
            (self.pieces[self.color][r_symb]).append(26)
            self.array[25:29] = [EMPTY_SQR, r_symb, k_symb, EMPTY_SQR]
        elif dest_idx == 23:  # white to queenside
            (self.pieces[self.color][k_symb]) = [23]
            (self.pieces[self.color][r_symb]).remove(21)
            (self.pieces[self.color][r_symb]).append(24)
            self.array[21:26] = [EMPTY_SQR, EMPTY_SQR, k_symb, r_symb,
                                  EMPTY_SQR]
        elif dest_idx == 97:  # black to kingside
            (self.pieces[self.color][k_symb]) = [97]
            (self.pieces[self.color][r_symb]).remove(98)
            (self.pieces[self.color][r_symb]).append(96)
            self.array[95:99] = [EMPTY_SQR, r_symb, k_symb, EMPTY_SQR]
        elif dest_idx == 93:  # black to queenside
            (self.pieces[self.color][k_symb]) = [93]
            (self.pieces[self.color][r_symb]).remove(91)
            (self.pieces[self.color][r_symb]).append(94)
            self.array[91:96] = [EMPTY_SQR, EMPTY_SQR, k_symb, r_symb,
                                  EMPTY_SQR]

    def is_king_missing(self) -> bool:
        return len(self.pieces[WHITE]['K']) == 0 or len(self.pieces[BLACK]['k']) == 0

    def remove_castling_right(self, cr) -> None:
        self.castling = self.castling.replace(cr, '')
        if len(self.castling) == 0:
            self.castling = '-'

    # Helpers-
    def is_sqr_friend(self, idx, perspective) -> bool:
        p = str(self.array[idx])
        if perspective == WHITE and p.isupper() or perspective == BLACK and p.islower():
            return True
        return False
    def is_sqr_enemy(self, idx, perspective) -> bool:
        p = str(self.array[idx])
        if perspective == WHITE and p.islower() or perspective == BLACK and p.isupper():
            return True
        return False
    def is_sqr_empty(self, idx) -> bool:
        return str(self.array[idx]) == EMPTY_SQR
    def is_sqr_illegal(self, idx) -> bool:
        return str(self.array[idx]) == ILLEGAL_SQR
    def is_sqr_threaten(self, idx, perspective) -> bool:
        ek, eq, er, en, eb, ep = 'kqrnbp' if perspective == WHITE else 'KQRNBP'
        p_pawn = 'P' if perspective == WHITE else 'p'
        k_dest = np.array(REG_MASKS[ek]) + idx
        if ek in [self.array[i] for i in k_dest]:
            return True
        n_dest = np.array(REG_MASKS[en]) + idx
        if en in [self.array[i] for i in n_dest]:
            return True
        p_dest = np.array(SPECIAL_MASKS[p_pawn]) + idx
        if ep in [self.array[i] for i in p_dest]:
            return True
        for m in REG_MASKS[er]:  # all vertical directions
            t_idx = idx + m
            while self.is_sqr_illegal(t_idx) is False:
                if self.array[t_idx] in (eq, er):
                    return True
                if self.is_sqr_empty(t_idx) is False:
                    break
                t_idx += m
        for m in REG_MASKS[eb]:  # all diagonal directions
            t_idx = idx + m
            while self.is_sqr_illegal(t_idx) is False:
                if self.array[t_idx] in (eq, eb):
                    return True
                if self.is_sqr_empty(t_idx) is False:
                    break
                t_idx += m
        return False

    def print_board(self) -> None:
        rc = 8
        r_factor = 0
        print(self.color, self.castling, self.en_passant, self.hm, self.fm)
        for rank in range(ARRAY_B_DIM):
            # r1, r2, r3, r4, r5, r6, r7, r8 = self.array[91 - r_factor: 98 - r_factor + 1]
            print("{}|{} {} {} {} {} {} {} {}".format(rc, *self.array[91 - r_factor: 98 - r_factor + 1]))
            r_factor += 10
            rc -= 1
        print("-----------------")
        print(" |a b c d e f g h")


##################################################################
#   MOVE:
##################################################################

class MoveGenerator:
    """ The MoveGenerator is the heart of the search process, and it must be quick and efficient:
            quick- calculate reachable squares fast.
            efficient- cancel illegal moves as soon as possible.
        The MoveGenerator is charge of ALL moves' validations !!
        The method generate_moves() takes a position on a PADDED_ARRAY_BOARD and generate a list of all possible
        moves (for all the player pieces)."""

    def __init__(self):
        pass

    def gen_moves(self, board: PaddedArrayBoard) -> list:
        # if board.is_king_missing():
        #     return []
        return self.gen_pseudo_legal(board)

    def gen_pseudo_legal(self, board) -> list:
        # TODO Must comprehend all types of moves (reg, castling, en-passant, capturing, promotion)-
        capture_moves = []
        reg_moves = []
        player_active_pieces = {k: v for (k,v) in board.pieces[board.color].items() if len(v) > 0}
        # regular moves-
        for p, idx_list in player_active_pieces.items():
            for i in idx_list:
                cptr, reg = None, None
                if p in ('P', 'p'):
                    cptr, reg = self.calc_pawn_moves(piece=p, p_idx=i, board=board)
                elif p in ('N', 'n'):
                    cptr, reg = self.calc_knight_moves(piece=p, p_idx=i, board=board)
                elif p in ('B', 'b'):
                    cptr, reg = self.calc_bishop_moves(piece=p, p_idx=i, board=board)
                elif p in ('R', 'r'):
                    cptr, reg = self.calc_rook_moves(piece=p, p_idx=i, board=board)
                elif p in ('Q', 'q'):
                    cptr, reg = self.calc_queen_moves(piece=p, p_idx=i, board=board)
                if p in ('K', 'k'):
                    cptr, reg = self.calc_king_moves(piece=p, p_idx=i, board=board)
                capture_moves += cptr
                reg_moves += reg
        # en passant-
        return capture_moves + reg_moves

    def calc_pawn_moves(self, piece, p_idx, board) -> tuple:  # calculates pseudo legal moves for a certain piece
        # TODO add en-passant move
        reg_mvs = []
        cptr_mvs = []
        m = REG_MASKS[piece]
        t_idx = p_idx + m  # target square index
        if board.is_sqr_illegal(t_idx) or board.is_sqr_friend(t_idx, perspective=board.color):  # out-of-bounds/wrap-around or fellow soldier
            return cptr_mvs, reg_mvs
        elif board.is_sqr_empty(t_idx):  # advance to empty target square
            m_nott = PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx]
            if t_idx // 10 == 9:  # add promotion appendix to move
                cptr_mvs += [m_nott + sym for sym in ['Q', 'R', 'N', 'B']]
            elif t_idx // 10 == 2:
                cptr_mvs += [m_nott + sym for sym in ['q', 'r', 'n', 'b']]
            else:
                cptr_mvs.append(m_nott)
            if PADDED_B_NOTATIONS[p_idx][1] in ('2', '7') and board.is_sqr_empty(t_idx + m):  # double-advance at start
                reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx + m])
        for m in SPECIAL_MASKS[piece]:  # capturing moves
            t_idx = p_idx + m
            if board.is_sqr_enemy(t_idx, perspective=board.color):
                m_nott = PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx]
                if t_idx // 10 == 9:  # add promotion appendix to move
                    cptr_mvs += [m_nott + sym for sym in ['Q', 'R', 'N', 'B']]
                elif t_idx // 10 == 2:
                    cptr_mvs += [m_nott + sym for sym in ['q', 'r', 'n', 'b']]
                else:
                    cptr_mvs.append(m_nott)
        return cptr_mvs, reg_mvs

    def calc_rook_moves(self, piece, p_idx, board) -> tuple:
        reg_mvs = []
        cptr_mvs = []
        for m in REG_MASKS[piece]:
            t_idx = p_idx + m
            while not board.is_sqr_illegal(t_idx):
                if board.is_sqr_friend(t_idx, perspective=board.color):
                    break
                elif board.is_sqr_enemy(t_idx, perspective=board.color):
                    cptr_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                    break
                elif board.is_sqr_empty(t_idx):
                    reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                t_idx += m
        return cptr_mvs, reg_mvs

    def calc_knight_moves(self, piece, p_idx, board) -> tuple:
        reg_mvs = []
        cptr_mvs = []
        for m in REG_MASKS[piece]:
            t_idx = p_idx + m
            if board.is_sqr_illegal(t_idx) or board.is_sqr_friend(t_idx, perspective=board.color):
                continue
            elif board.is_sqr_empty(t_idx):
                reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
            elif board.is_sqr_enemy(t_idx, perspective=board.color):
                cptr_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
        return cptr_mvs, reg_mvs

    def calc_bishop_moves(self, piece, p_idx, board) -> tuple:
        reg_mvs = []
        cptr_mvs = []
        for m in REG_MASKS[piece]:
            t_idx = p_idx + m
            while not board.is_sqr_illegal(t_idx):
                if board.is_sqr_friend(t_idx, perspective=board.color):
                    break
                elif board.is_sqr_enemy(t_idx, perspective=board.color):
                    cptr_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                    break
                elif board.is_sqr_empty(t_idx):
                    reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                t_idx += m
        return cptr_mvs, reg_mvs

    def calc_queen_moves(self, piece, p_idx, board) -> tuple:
        reg_mvs = []
        cptr_mvs = []
        for m in REG_MASKS[piece]:
            t_idx = p_idx + m
            while not board.is_sqr_illegal(t_idx):
                if board.is_sqr_friend(t_idx, perspective=board.color):
                    break
                elif board.is_sqr_enemy(t_idx, perspective=board.color):
                    cptr_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                    break
                elif board.is_sqr_empty(t_idx):
                    reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
                t_idx += m
        return cptr_mvs, reg_mvs

    def calc_king_moves(self, piece, p_idx, board) -> tuple:
        reg_mvs = []
        cptr_mvs = []
        for m in REG_MASKS[piece]:
            t_idx = p_idx + m
            if board.is_sqr_illegal(t_idx) or board.is_sqr_friend(t_idx, perspective=board.color):
                continue
            elif board.is_sqr_empty(t_idx):
                reg_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
            elif board.is_sqr_enemy(t_idx, perspective=board.color):
                cptr_mvs.append(PADDED_B_NOTATIONS[p_idx] + PADDED_B_NOTATIONS[t_idx])
        for r in set(board.castling).intersection(PLAYER_RIGHTS[board.color]):
            low, high = CASTLING_PATH_IDX[r]
            if board.array[low: high] == CASTLING_EMPTY_PATH[r]:
                reg_mvs.append(CASTLING_MOVES[r])
        return cptr_mvs, reg_mvs


##################################################################
#   EVALUATION:
##################################################################

class Evaluator:
    """ This class holds functionality for score evaluation for a board position
        This class evaluate score for a Zero-Sum-Game, i.e. player1 gains what player2 loses.
        Calculations must be light, for minimum burden on the mini-max algorithm.
        Moreover, evaluation must detect Check and Checkmate as highly rated moves.
        TODO consider CI heuristic methods
    """

    def __init__(self):
        pass

    def evaluate(self, board: PaddedArrayBoard, maxi_color) -> int:
        enm_color = BLACK if maxi_color == WHITE else WHITE
        score = 0
        # Material eval:
        for p, mvs in board.pieces[maxi_color].items():
            score += PIECES_VALUES[p.upper()] * len(mvs)
        for p, mvs in board.pieces[enm_color].items():
            score -= PIECES_VALUES[p.upper()] * len(mvs)

        # PST eval:
        # TODO - add pst in the future

        return int(score)

    def calc_depth_penalty(self, d: int):
        BASE_FACTOR = 10
        return BASE_FACTOR * d


##################################################################
#   SEARCH:
##################################################################

class SmartMovesDict:
    def __init__(self):
        self.moves = SortedDict()

    def insert(self, score, move):
        if score in self.moves.keys():
            self.moves[score].append(move)
        else:
            self.moves[score] = [move]

    def get(self, score):
        if len(self.moves) == 0:
            return NULL_MOVE
        b_moves = self.moves[score]
        if len(b_moves) > 1:
            return np.random.choice(b_moves)
        else:
            return b_moves[0]


class Searcher:
    """ This class uses a minimax algorithm with alpha-beta pruning,
        to build a search tree and search for optimal next move"""
    DEF_MAX_DEPTH = 3
    DEF_MAX_TIME = sys.maxunicode

    def __init__(self):
        # Helper classes:
        self.generator = MoveGenerator()
        self.evaluator = Evaluator()
        # Search process params:
        self.max_depth = self.DEF_MAX_DEPTH     # define leaves by depth
        self.max_time = self.DEF_MAX_DEPTH      # define leaves by time
        self.max_player = ''
        self.best_moves = None
        self.nodes = 0

    def search_minimax(self, board: PaddedArrayBoard, depth=DEF_MAX_DEPTH, time=DEF_MAX_TIME) -> str:
        # A minimax search algorithm
        self.max_depth = depth
        self.max_time = time
        self.max_player = board.color
        self.best_moves = SmartMovesDict()
        self.nodes = 0
        t_start = timer.time()
        opt_score = self.find_max(board=board, depth=0,
                                  alpha=-1*PIECES_VALUES["HIGHEST"],
                                  beta=PIECES_VALUES["HIGHEST"])
        t_fin = timer.time()
        res = self.best_moves.get(opt_score)
        print(f'Search stats: player={self.max_player}\tdepth={self.max_depth}'
              f'\tbest_move={res}\tscored={opt_score}\tnodes={self.nodes}\t'
              f'time={round(t_fin-t_start, 5)}')
        return res

    def find_max(self, board: PaddedArrayBoard, depth, alpha, beta) -> int:
        self.nodes += 1
        if depth == self.max_depth or board.is_king_missing():     # leaf conditions
            return self.evaluator.evaluate(board, self.max_player)
        possible_moves = self.generator.gen_moves(board)
        v = -1 * PIECES_VALUES["HIGHEST"]  # worst case for max
        for mv in possible_moves:
            tb = board.make_move(move=mv)
            if tb.is_sqr_threaten(tb.pieces[board.color]['K' if board.color == WHITE else 'k'][0], board.color):
                """ throw away moves that conclude
                    in self-check """
                continue
            score = self.find_min(tb, depth + 1, alpha, beta)
            if depth == 0:  # for depth 0 (maxi_p) - save moves by best scores
                self.best_moves.insert(score, mv)
            if v < score:  # get maximum successor (out of minimums)
                v = score
            if beta <= v:  # pruning
                return v
            alpha = max([alpha, v])
        return v

    def find_min(self, board: PaddedArrayBoard, depth, alpha, beta) -> int:
        self.nodes += 1
        if depth == self.max_depth or board.is_king_missing():
            return self.evaluator.evaluate(board, self.max_player)
        possible_moves = self.generator.gen_moves(board)
        v = PIECES_VALUES["HIGHEST"]
        for mv in possible_moves:
            tb = board.make_move(move=mv)
            if tb.is_sqr_threaten(tb.pieces[board.color]['K' if board.color == WHITE else 'k'][0], board.color):
                continue
            score = self.find_max(tb, depth + 1, alpha, beta)
            if score < v:
                v = score
            if v <= alpha:  # pruning
                return v
            beta = min([beta, v])
        return v


##################################################################
#   ENGINE INTERFACE:
##################################################################

class UCI:
    F_DEBUG = False
    F_READY = False
    F_PONDER = False
    F_STOP = False
    def __init__(self):
        self.board = None
        self.searcher = None
        self.search_max_depth = 4

    def run_interface(self):
        while True:
            my_io = input()
            if my_io == "uci":
                self.id()
                self.declare_options()
                self.uciok()
            elif my_io.startswith("debug"):
                # set the engine debug mode. on = True, off = False
                my_io = my_io[len("debug "):]
                self.debug(my_io)
            elif my_io == "isready":
                # gui is ready, engine can setup.
                self.isready()
            elif my_io.startswith("setoption name"):
                # this command is sent when the user wants to change the engine parameters
                my_io = my_io[len("setoption name "):]
                self.setoption(my_io)
            elif my_io == "register":
                # try to register the engine, or tell it that it would be done later.
                self.register()
            elif my_io == "ucinewgame":
                # sent before uplaoding a position from a different game
                self.ucinewgame()
            elif my_io.startswith("position"):
                # set up the board according to a position described in fen-string.
                my_io = my_io[len("position "):]
                self.position(my_io)
            elif my_io.startswith("go"):
                # start calculating on the current position of the board.
                my_io = my_io[len("go "):]
                self.go(my_io)
                #t = Thread(target=self.go, args=my_io, daemon=True)
                #t.start()
            elif my_io == "stop":
                # stop calculating as soon as possible.
                # Do provide 'best_move' and possibly 'ponder' tokens.
                self.stop()
                #t = Thread(target=self.stop(), args=my_io, daemon=True)
                #t.start()
            elif my_io == "ponderhit":
                self.ponderhit()
            elif my_io == "quit":
                # quit program as soon as possible.
                return

    # Engine-to-GUI methods:
    def id(self):
        print('id name SkyFioli')
        print('id author Eran Avraham')
    def declare_options(self):
        pass
    def uciok(self):
        print('uciok')

    # GUI-to-ENGINE methods:
    def debug(self, params) -> None:
        pass

    def isready(self) -> None:
        print('readyok')

    def setoption(self, params) -> None:
        pass

    def register(self) -> None:
        pass

    def ucinewgame(self) -> None:
        self.board = PaddedArrayBoard()
        self.searcher = Searcher()

    def position(self, params) -> None:
        params = params.split()
        fen_pos = params[0]

        self.board.parse_fen(FEN_START if fen_pos == 'startpos' else fen_pos)
        if len(params) > 2:
            moves = params[params.index('moves') + 1:]
            for m in moves:
                self.board = self.board.make_move(m)

    def go(self, param_string) -> None:
        if param_string != '':
            pass
        # pool = ThreadPool(processes=1)
        # async_result = pool.apply_async(self.searcher.search_minimax, args=[self.board])  # tuple of args for foo
        # best_mv = async_result.get()
        best_mv = self.searcher.search_minimax(self.board, depth=self.search_max_depth)
        print(f'bestmove {best_mv}')

    def stop(self) -> None:
        print(f'engine stopping..')
        self.F_STOP = True

    def ponderhit(self) -> None:
        pass


class DEVELOPER:
    def __init__(self):
        self.board = None

    def run_interface(self):
        self.board = PaddedArrayBoard()
        evaluator = Evaluator()
        print('Welcome Developer!')
        while True:
            my_io = input()
            if my_io.startswith("thread print"):
                my_io = my_io[len("thread print "):]
                t = Thread(target=print, args=my_io)
                t.run()
            elif my_io == "new board":
                self.board = PaddedArrayBoard()
            elif my_io.startswith("set"):
                my_io = my_io[len("set "):]
                self.board.parse_fen(fen_str=my_io)
            elif my_io == "gen moves":
                generator = MoveGenerator()
                res = generator.gen_moves(self.board)
                print("moves found:", len(res))
                print(res)
            elif my_io == "eval":
                res = evaluator.evaluate(self.board, self.board.color)
                print(res)
            elif my_io == "srch":
                searcher = Searcher()
                found_move = searcher.search_minimax(self.board)
                print(self.board.color, found_move)
            elif my_io.startswith("move"):
                my_io = my_io[len("move "):]
                self.board = self.board.make_move(move=my_io)
                self.board.print_board()
            elif my_io.startswith("thrt"):
                idx, plyr = my_io[len("thrt "):].split()
                idx = int(idx)
                print(self.board.is_sqr_threaten(idx, plyr))
            elif my_io.startswith("print"):
                my_io = my_io[len("print "):]
                if my_io == "brd" and self.board.__class__ == PaddedArrayBoard:
                    self.board.print_board()
                elif my_io == "pcs":
                    print(self.board.pieces[self.board.color])
                    print(self.board.pieces[self.board.enemy])
                elif my_io == 'fen':
                    print(self.board.get_fen())
            elif my_io == "run game":
                print(">>>\tStarting game simulation")
                print(">>>\tPlayer: w\tComputer: b")
                searcher = Searcher()
                for i in range(10):
                    self.board.print_board()
                    arsenal = searcher.generator.gen_moves(self.board)
                    print(">>>\tYour possible moves:", len(arsenal), arsenal)
                    mv = input(">>>\tWhite move:")
                    if mv == "exit":
                        return
                    while mv not in arsenal:
                        mv = input(">>>\tWhite move:")
                    self.board = self.board.make_move(move=mv)  # player's move

                    self.board.print_board()
                    mv = searcher.search_minimax(self.board)
                    print(">>>\tBlack move:", mv)
                    self.board = self.board.make_move(move=mv)  # bot move
            elif my_io == "run bots":
                ROUNDS = 50
                print(">>>\tStarting 2 bots simulation")
                print(">>>\tBot #1: w\tBot #2: b")
                searcher = Searcher()
                for i in range(ROUNDS):
                    mv = searcher.search_minimax(self.board, depth=3)
                    if mv == NULL_MOVE:
                        print('>>>\tGame Over! no moves')
                        print(f'Winner: {self.board.enemy}')
                        print(self.board.get_fen())
                        self.board.print_board()
                        return
                    else:
                        print(f'#{i + 1}\tplr={self.board.color}\tmove={mv}')
                        self.board = self.board.make_move(move=mv)
            elif my_io == "test":
                print(">>>\tStarting Test")
                searcher = Searcher()

                tests = (FEN_START, FEN_TRICKY_TEST, FEN_KILLER_POSITION,
                         FEN_PROMOTION_TEST, FEN_ENDGAME_TEST)
                rounds = 5
                half_turns = 50
                for fen in tests:
                    self.board.parse_fen(fen)
                    for rnd in range(rounds):
                        print(f'>>> Rond {rnd} for board {fen}')
                        self.board.print_board()
                        for i in range(half_turns):
                            mv = searcher.search_minimax(self.board, depth=4)
                            if mv == NULL_MOVE:
                                print(f'>>> Round {rnd} Game Over!')
                                print(f'>>> Round {rnd} Winner: {self.board.enemy}')
                                self.board.print_board()
                                self.board.parse_fen(fen)
                                break
                            else:
                                print(f' {self.board.color}\tmove={mv}\tfen={self.board.get_fen()}')
                                self.board = self.board.make_move(move=mv)
            else:
                print("Unknown command")


if __name__ == "__main__":
    uci = UCI()
    uci.run_interface()
