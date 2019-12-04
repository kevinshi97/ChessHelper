import numpy as np
import cv2

'''
This file contains all the code necessary for preprocessing a screenshot
'''

def get_image(image_path):
    '''
    load the image from the image path and resizes the image.
    '''
    image = cv2.imread(image_path)

    # here we assume that the image is going to be roughly square or that it will be wider than it is longer
    width = 1000
    scale = width/float(image.shape[1])
    height = int(image.shape[0] * scale)
    image = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

    cv2.imwrite('out/screenshot.jpg', image)
    
    return image

def get_lines_from_image(image, threshold = 210):
    '''
    load the image, calculates the canny edge. Take the canny edge and perform hough lines to find the prominent lines in the image
    '''
    line_image = image.copy()
    edges = cv2.Canny(line_image,50,150,apertureSize = 3)
    cv2.imwrite('out/cannyedge.jpg', edges)

    lines = []
    hough_lines = cv2.HoughLines(edges,1,np.pi/180, threshold)
    all_lines = image.copy()
    for hline in hough_lines:
        for rho,theta in hline:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            lines.append(((x1, y1), (x2, y2))) 
            cv2.line(all_lines,(x1,y1),(x2,y2),(0,0,255),1)

    cv2.imwrite('out/houghline.jpg', all_lines)
    return lines


def get_lines_by_orientation(lines):
    '''
    lines denoted as: ((x1, y1), (x2, y2)) where (x1, y1) is a point and (x2, y2) is another point
    return two lists of lines depending on their orientation
    '''
    horizontals = []
    verticals = []
    epsilon = 5           # allows room for error by 5 pixels
    for line in lines:
        x1,y1,x2,y2 = line[0][0], line[0][1], line[1][0], line[1][1]
        # since all lines span the entirety of the image, we only need they y values for vertical lines and x values for horizontal lines
        # check slopes of lines
        if abs(y1 - y2) < epsilon:
            horizontals.append(y1)
        elif abs(x1 - x2) < epsilon:
            verticals.append(x1)
        else:
            del line

    horizontals.sort()
    verticals.sort()
    return horizontals, verticals

def prune_lines(lines):
    '''
    Prunes a set of lines to 7 in consistent increasing order (chessboard). Keep the lines that are evenly spaced apart
    '''
    diff = np.diff(lines)
    x, count = 0, 0
    epsilon = 5

    start = 0
    for i, line in enumerate(diff):
        # Within 5 px of the other (allowing for minor image errors)
        if np.abs(line - x) < epsilon:
            count += 1
            if count == 7:
                end = i + 2
                return lines[start:end]
        else:
            count = 0
            x = line
            start = i
    return lines
    # return None

def get_chessboard(image, threshold = 250):
    '''
    Function called in main.py that will take the image and returns the board with the x and y indices for the board. There should be
    eight x_indices and eight y_indices. The looping through the x_indices and y_indices should create 64 patches.
    '''
    lines = get_lines_from_image(image, threshold = threshold)
    horizontals, verticals = get_lines_by_orientation(lines)
    board_verticals = prune_lines(verticals)
    board_horizontals = prune_lines(horizontals)

    start_x, end_x, start_y, end_y = board_verticals[0], board_verticals[-1], board_horizontals[0], board_horizontals[-1]
    board = image[start_y:end_y, start_x:end_x]
    x_indices = [x - start_x for x in board_verticals]    #indices relative to the new boards indices
    y_indices = [y - start_y for y in board_horizontals]
    cv2.imwrite('out/board.jpg', board)

    return board, x_indices, y_indices

def get_patches(board, x_indices, y_indices):
    '''
    get the board and the indices to generate patches. The looping through the x_indices and y_indices should create 64 patches.
    '''
    patches = []
    for i in range(len(y_indices) - 1):
        for j in range(len(x_indices) - 1):
            patches.append(board[y_indices[i]: y_indices[i + 1], x_indices[j]: x_indices[j + 1], :])
    return patches