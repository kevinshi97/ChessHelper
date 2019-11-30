from scipy.spatial import distance
from operator import itemgetter, attrgetter
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

IMAGE_SIZE = 800
BOARD_SIZE = 8
WINDOW_NAME = 'image'

EMPTY = 0
PAWN = 1
ROOK = 2
KNIGHT = 3
BISHOP = 4
QUEEN = 5
KING = 6


# board = [[' ' for i in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
board = [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
        ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
        ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]

def board_to_fen(board, turn = 'w'):
    fen = ''
    for i in range(BOARD_SIZE):
        empty = 0
        for j in range(BOARD_SIZE):
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

def visualize_keypoints(image, keypoints, radius = 3, colour = (0, 0, 255), thickness = -1):
    print(image.shape)
    print(keypoints)
    for keypoint in keypoints:
        print(keypoint)
        cv2.circle(image, keypoint, radius, colour, thickness) 
    return image


def transform_image(image, keypoints):
    cv2.imshow('visualize keypoints', visualize_keypoints(image, keypoints))

    transform_keypoints = [(0,0),(IMAGE_SIZE,0),(0,IMAGE_SIZE),(IMAGE_SIZE,IMAGE_SIZE)]
    M = cv2.getPerspectiveTransform(np.float32(keypoints),np.float32(transform_keypoints))
    return cv2.warpPerspective(board_image,M,(IMAGE_SIZE, IMAGE_SIZE))

def visualize_grid(image):
    for i in range(0,IMAGE_SIZE+1,int(IMAGE_SIZE/float(BOARD_SIZE))):
        cv2.line(image,(0, i),(IMAGE_SIZE, i),(0,0,255),2)
        cv2.line(image,(i, 0),(i, IMAGE_SIZE), (0,0,255),2)
    return image

def mySiftMatch(I, J, norm = 'euclidean', threshold = 0.8):
    print("SIFT with ", norm, "norm and a threshold of ", threshold)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints_I, descriptors_I = sift.detectAndCompute(I, None)
    keypoints_J, descriptors_J = sift.detectAndCompute(J, None)

    matches = []
    for i in range(len(keypoints_I)): 
        distances = distance.cdist([descriptors_I[i]], descriptors_J, norm)
        distances = distances[0]
        keypoint_dist = []
        for j in range(len(keypoints_J)):
            keypoint_dist.append([j, distances[j]])

        keypoint_dist = sorted(keypoint_dist, key = itemgetter(1))
        if keypoint_dist[0][1]/(keypoint_dist[1][1]) < threshold:
            matches.append(cv2.DMatch(i, keypoint_dist[0][0], keypoint_dist[0][1]))

    print('all matches:', len(matches))
    return len(matches), matches, keypoints_I, keypoints_J

def patch_sift(I):
    pieces = ['pawn', 'rook', 'knight', 'bishop', 'queen', 'king']
    piece_images = []
    for piece in pieces:
        filepath = './assets/training_images/pieces/'+ piece +'/no_board/'
        for file in os.listdir(filepath):
            if 1:
                img = cv2.imread(filepath + file)
                img = cv2.resize(img, (img.shape[1]//5,img.shape[0]//5))
                piece_images.append(img)


    patch_size = int(IMAGE_SIZE/float(BOARD_SIZE))
    for i in range(0,IMAGE_SIZE+1-patch_size, patch_size):
        for j in range(0,IMAGE_SIZE+1-patch_size, patch_size):
            max_num = 0
            max_matches, max_keypoints_I, max_keypoints_J = [], [], []
            max_img = None
            patch = I[i:i + patch_size, j:j + patch_size, :]
            for img in piece_images:
                num, matches, keypoints_I, keypoints_J = mySiftMatch(patch, img, threshold = 0.6)
                if num > max_num:
                    max_num = num
                    max_matches, max_keypoints_I, max_keypoints_J = matches, keypoints_I, keypoints_J
                    max_img = img
            print('most matches:', max_num)
            if max_num > 0:
                cv2.imshow('matches', cv2.drawMatches(patch, max_keypoints_I, max_img, max_keypoints_J, max_matches, None, flags=2))
                cv2.waitKey(0)
            else:
                print('empty')
                



if __name__ == "__main__":
    # board_image = cv2.imread('/content/gdrive/My Drive/Colab Notebooks/project/assets/training_images/board/no_projection/board45-1.jpg')
    # board_keypoints = [(320, 1058), (1490, 1067), (112, 2144), (1696, 2138)]
    
    # board_image = cv2.imread('/content/gdrive/My Drive/Colab Notebooks/project/assets/training_images/board/no_projection/board80-1.jpg')
    # board_keypoints = [(299, 1054), (1402, 1067), (233, 2192), (1485, 2174)]   
    # board_keypoints = [(355, 1110), (1346, 1121), (304, 2120), (1414, 2109)] 
    
    board_image = cv2.imread('./assets/training_images/board/no_projection/board90-3.jpg')
    board_keypoints = [(460, 1143), (1443, 1143), (460, 2142), (1463, 2128)]  
    # plt.imshow(board_image)
    # plt.show()
    
    board_transformed = transform_image(board_image, board_keypoints)
    cv2.imshow('board_transform', board_transformed)
    cv2.waitKey(0)

    # img = cv2.imread('sudokusmall.png')
    # rows,cols,ch = img.shape

    # pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
    # pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

    # M = cv2.getPerspectiveTransform(pts1,pts2)

    # dst = cv2.warpPerspective(img,M,(300,300))

    # plt.subplot(121),plt.imshow(img),plt.title('Input')
    # plt.subplot(122),plt.imshow(dst),plt.title('Output')
    # plt.show()



    # board_image = cv2.imread('/content/gdrive/My Drive/Colab Notebooks/project/assets/training_images/board/9.jpg')
    # board_image = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    board_grid_image = visualize_grid(board_transformed)
    cv2.imshow('board_grid_image', board_grid_image)
    cv2.waitKey(0)


    # matches, board_keypoints, bishop_keypoints = mySiftMatch(board_transformed, bishop_image, threshold = 0.6)

    # cv2_imshow(cv2.drawMatches(board_transformed, board_keypoints, bishop_image, bishop_keypoints, matches, None, flags=2))
    # cv2.waitKey(0)
    patch_sift(board_transformed)

    print(board_to_fen(board))
    # cv2.imwrite('/content/gdrive/My Drive/Colab Notebooks/project/assets/training_images/board/houghlines3.jpg',img)


