##
# Date: 2/22/2020
# Author: Lei Yang
# Description: Project 1 -- Feature Matching + Homography to find Objects
# Reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography
##
import numpy as np
import matplotlib.pyplot as plt
import cv2

# image reading and adjust channel
img1 = cv2.imread('object.png')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('scene.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# calculate keypoint for each image
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# use FLANN matching
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 指定递归次数
flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)  # k=2 find the 2 nearest point
goodMatch = []
# store all the good matches as per Lowe's ratio test. If misfit of the first matching point is 70% smaller than another matching point, it is a good match
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatch.append(m)

# find homography matrix using RANSAC and do the perspective transform
src_pts = np.float32([kp1[m.queryIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in goodMatch]).reshape(-1, 1, 2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
matchesMask = mask.ravel().tolist()  # turn ndarray to list
h, w, c = img1.shape
pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
dst = cv2.perspectiveTransform(pts, M)

# generate the polygonal curves of the tranformed object in the scene
img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
draw_params = dict(matchColor=(0, 0, 255), singlePointColor=(0, 0, 255), matchesMask=matchesMask, flags=2)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, goodMatch, None, **draw_params)  
plt.imshow(img3)
plt.show()
