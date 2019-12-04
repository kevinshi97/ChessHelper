import cv2
import torch
import numpy as np

'''
Dictionary of CNN numerical classes to fen strings.
bishop: 0 -> b
empty: 1 -> ' '
king: 2 -> k
knight: 3 -> n
pawn: 4 -> p
queen: 5 -> q
rook: 6 -> r
'''
piece_class = {'0': 'b', '1': ' ', '2': 'k', '3': 'n', '4': 'p', '5': 'q', '6': 'r'}

def get_positions(piece_predictions):
    '''
    Given the list of piece predictions, place them on a 2-D array. 
    '''
    num_board = np.array(piece_predictions).reshape(8, 8)
    fen_board = [[' ' for i in range(num_board.shape[1])] for i in range(num_board.shape[0])]
    print(num_board)
    for i in range(num_board.shape[0]):
        for j in range(num_board.shape[1]):
            fen_board[i][j] = piece_class[str(int(num_board[i, j]))]
    return fen_board

def get_color(patches, fen_board):
    '''
    takes a list of patches and returns the 2D board
    '''
    for i in range(len(patches)):
      gray = cv2.cvtColor(patches[i], cv2.COLOR_BGR2GRAY)
      resize = cv2.resize(gray, (64, 64), interpolation = cv2.INTER_AREA)
      index = np.ix_(range(12, 50),range(12, 50))
      cropped = resize[index]
      cropped[cropped > 220] = 255
      if np.average(cropped) > 129:
          fen_board[i//8][i%8] = fen_board[i//8][i%8].upper()
    return fen_board

def board_to_fen(board, turn = 'w'):
    '''
    Given a board state and a turn, returns the fen.
    '''
    fen = ''
    for i in range(len(board)):
        empty = 0
        for j in range(len(board[i])):
            square = board[i][j]
            if square != ' ':
                if empty > 0:
                    fen += str(empty)
                    empty = 0
                fen += square
            else:
                empty += 1
        if empty > 0:
            fen += str(empty)
        fen += '/'

    fen = fen[:-1]
    fen += ' %s - - 0 1'%(turn)

    return fen
