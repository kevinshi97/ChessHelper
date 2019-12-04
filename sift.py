from scipy.spatial import distance

'''
The code in this file is all of the sift stuff we had before we decided to scrap it. Sift was barely detecting any very good features and the matching
was all over the place. We decided not to include it in our final product but the code is still here if you want to take a look. Unfortunatly it will 
almost definetly fail to run if you try to use it since we moved on from it a while ago.
'''
# ---------------------------------------------------------------------------------------------------------------------------------------------

# this was for sift, which we decided not to do
# def load_train_data():
#     pieces_dict= {'pawn': [], 'rook': [], 'knight': [], 'bishop': [], 'queen': [], 'king': []}
#     for piece, images in pieces_dict.items():
#         filepath = '/content/gdrive/My Drive/Colab Notebooks/project/assets/train/'+ piece +'/'
#         for file in os.listdir(filepath):
#             img = cv2.imread(filepath + file)
#             img = cv2.resize(img, (img.shape[1]//5,img.shape[0]//5))
#             images.append(img)
#     return pieces_dict

# def mySiftMatch(I, J, norm = 'euclidean', threshold = 0.8):
#     # print("SIFT with ", norm, "norm and a threshold of ", threshold)
#     sift = cv2.xfeatures2d.SIFT_create()
#     keypoints_I, descriptors_I = sift.detectAndCompute(I, None)
#     keypoints_J, descriptors_J = sift.detectAndCompute(J, None)

#     matches = []
#     for i in range(len(keypoints_I)): 
#         distances = distance.cdist([descriptors_I[i]], descriptors_J, norm)
#         distances = distances[0]
#         keypoint_dist = []
#         for j in range(len(keypoints_J)):
#             keypoint_dist.append([j, distances[j]])

#         keypoint_dist = sorted(keypoint_dist, key = itemgetter(1))
#         if keypoint_dist[0][1]/(keypoint_dist[1][1]) < threshold:
#             matches.append(cv2.DMatch(i, keypoint_dist[0][0], keypoint_dist[0][1]))

#     # print('all matches:', len(matches))
#     return len(matches), matches, keypoints_I, keypoints_J

# def sift_on_patches(patches, pieces_dict):
#     for patch in patches:
#         for piece, images in pieces_dict.items():
#             for piece_image in images:
#                 patch_sift = patch.copy()
#                 piece_sift = piece_image.copy()
#                 num, matches, keypoints_I, keypoints_J = mySiftMatch(patch_sift, piece_sift, threshold = 0.80)

#                 patch_sift = cv2.drawKeypoints(patch_sift,keypoints_I,patch_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 piece_sift = cv2.drawKeypoints(piece_sift,keypoints_J,piece_sift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#                 cv2_imshow(cv2.drawMatches(patch_sift, keypoints_I, piece_sift, keypoints_J, matches, None, flags=2))
# ---------------------------------------------------------------------------------------------------------------------------------------------