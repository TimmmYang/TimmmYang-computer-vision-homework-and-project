## Project 1 -- Feature Matching + Homography to find Objects ##



This is the project 1 of computer vision course. The project is implemented in the following step:

1. Feature extraction. Using OpenCV SIFT algorithm to extract the features of the keypoints in the object and the scene.
2. Feature matching. Using FLANN (Fast Library for Approximate Nearest Neighbors) library to match keypoints. Specifically, KNN (K-nearest neighbors) is applied in the code.
3. Find homography matrix using RANSAC (Random Sample Consensus) method and do the perspective transform.
4. Generate the polygonal curves of the tranformed object in the scene and plot the figure.



References:

https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#feature-homography

https://blog.csdn.net/HuangZhang_123/article/details/80660688
