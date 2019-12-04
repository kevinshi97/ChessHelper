import cv2
import torch
import numpy as np
from argparse import ArgumentParser

from stockfish import Stockfish
from model import SimpleNet, load_input, train, predict_piece
from preprocess import get_image, get_chessboard, get_patches

'''
command line arguments:
-turn or -t specifies the whos turn it is. Use b or w to specify if it is black or white's move
-path or -p specifies the image path. there are a few images included in test for you to run without any tuning.
'''

parser = ArgumentParser(description="Bufferbloat tests")
parser.add_argument('--turn', '-t',
                    type=str,
                    help="who's turn is it? (either b/w)",
                    default='w')

parser.add_argument('--path', '-p',
                    type=str,
                    help="the images filepath",
                    required=True)

args = parser.parse_args()

piece_class = {'0': 'b', '1': ' ', '2': 'k', '3': 'n', '4': 'p', '5': 'q', '6': 'r'}

def get_positions(piece_predictions):
    num_board = np.array(piece_predictions).reshape(8, 8)
    fen_board = [[' ' for i in range(num_board.shape[1])] for i in range(num_board.shape[0])]
    # print(num_board)
    for i in range(num_board.shape[0]):
        for j in range(num_board.shape[1]):
            fen_board[i][j] = piece_class[str(int(num_board[i, j]))]
    return fen_board

def board_to_fen(board, turn = 'w'):
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

def get_color(patches, fen_board):
    for i in range(len(patches)):
      gray = cv2.cvtColor(patches[i], cv2.COLOR_BGR2GRAY)
      resize = cv2.resize(gray, (64, 64), interpolation = cv2.INTER_AREA)
      index = np.ix_(range(12, 50),range(12, 50))
      cropped = resize[index]
      cropped[cropped > 220] = 255
      if np.average(cropped) > 129:
          fen_board[i//8][i%8] = fen_board[i//8][i%8].upper()
    return fen_board

if __name__ == "__main__":
    model = SimpleNet(7)
    model.load_state_dict(torch.load('assets/model.h5'))

    '''
    The commented out section below is used to load the training data, train the model, and save the model. Since we already did
    all of that, we commented it out. Feel free to uncomment this to see it work.
    '''
    # training_set = load_input('assets/train')
    # train(model, training_set, epochs = 25)
    # torch.save(model.state_dict(), 'assets/model.h5')

    image = get_image(args.path)

    board, x_indices, y_indices = get_chessboard(image, threshold = 280)

    cv2.imwrite('out/board.jpg', board)

    patches = get_patches(board, x_indices, y_indices)
    piece_predictions = predict_piece(model, patches)
    fen_board = get_positions(piece_predictions)

    fen_board = get_color(patches, fen_board)
    print(np.array(fen_board))
    fen = board_to_fen(fen_board, turn = args.turn)
    print(np.array(fen))

    '''
    The sift stuff we had before we decided to scrap it. Look at sift.py to see what it did. The code below was written in colab and most likely
    will not run anymore; it is just here to look at.
    '''
    # pieces_dict = load_train_data()
    # patches = get_patches(board, x_indices, y_indices)
    # sift_on_patches(patches, pieces_dict)
    # cv2_imshow(board_image)

    # you should install the stockfish engine in your operating system globally or specify path to binary file in class constructor
    stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x64.exe')  # for windows 64 bit
    # stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x32.exe')  # for windows 32 bit
    # stockfish = Stockfish('assets/stockfish/Mac/stockfish-10-64')           # for mac 64 bit
    # stockfish = Stockfish('assets/stockfish/Linux/stockfish_10_x64')        # for linux 64 bit

    # set position by FEN:
    stockfish.set_fen_position(fen)
    print("The best move is: " + stockfish.get_best_move())
