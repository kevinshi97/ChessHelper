from argparse import ArgumentParser
import cv2
import torch
import numpy as np

from stockfish import Stockfish
from model import SimpleNet, load_input, train, predict_piece
from preprocess import get_image, get_chessboard, get_patches
from postprocess import get_positions, board_to_fen, get_color


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
