import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import pickle
# from IPython.display import display, Latex

PATH = os.path.join(os.getcwd(), os.pardir, 'data', 'Chessboard dataset')
# Define the dimensions of the chessboard pattern
chessboard_size = (7, 5)
# Define the criteria for subpixel accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare the object points
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Define the arrays to store the object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane

images = glob.glob(os.path.join(PATH, '*.jpg'))

# This code calibrates a camera using a set of images of a chessboard pattern.
# The camera matrix and distortion coefficients are calculated.
# We then draw the corners of the chessboard pattern on the image and display them.

# Find the chessboard corners in each image
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('image', img)
        cv2.waitKey(300)

cv2.destroyAllWindows()

# Calibrate the camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1])

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
        cv2.imshow('image', img)
        cv2.waitKey(300)

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1])

print(f'Focal length (fx, fy): {mtx[0, 0]}, {mtx[1, 1]}')
print(f'Skew parameter: {mtx[0, 1]}')
print(f'Principal point (cx, cy): {mtx[0, 2]}, {mtx[1, 2]}')
print(f'Error estimates: {ret}')
print(f'Camera matrix: \n{mtx}')

R = np.zeros((len(rvecs), 3, 3))
for i in range(len(rvecs)):
    R[i] = cv2.Rodrigues(rvecs[i])[0]

print(f'Rotation matrix: \n{R}')

print(f'Translation vector: \n{tvecs}')

print("Estimated radial distortion coefficients:")
print(dist)

raw_images = ['01', '02', '03', '04', '05']
for i in raw_images:

    img = cv2.imread(glob.glob(os.path.join(PATH, f'*{i}.jpg'))[0])
    undistorted = cv2.undistort(img, mtx, dist)
    
    cv2.imshow('Original', img)
    cv2.imshow('Undistorted', undistorted)
    cv2.waitKey(0)
cv2.destroyAllWindows()

mean_error = 0
errors = []
for i in range(len(objpoints)):

    # Find the error of the image points by projecting the object points with the 
    # rotation and translation vectors and comparing the result to the image points.
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    # error = np.sqrt(np.sum((imgpoints[i] - imgpoints2)**2) / len(imgpoints2))

    errors.append(error)
    mean_error += error
    print(f'Re-projection error for image {i+1}: {error}')

mean_error /= len(objpoints)
std_error = np.std(errors)
print(f'Mean re-projection error: {mean_error}')
print(f'Standard deviation of re-projection error: {std_error}')

plt.figure()
plt.bar(range(len(errors)), errors)
plt.xlabel('Image')
plt.ylabel('Re-projection error')
plt.title('Re-projection errors for camera calibration')
plt.show()

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret == True:
                
        cv2.drawChessboardCorners(img, chessboard_size, corners, bool(ret))
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1])
        imgpoints2, _ = cv2.projectPoints(objp, rvecs[0], tvecs[0], mtx, dist)
        
        cv2.drawChessboardCorners(img, chessboard_size, imgpoints2, bool(ret))
        
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Detected corners')
        axs[0].plot(corners[:, 0, 0], corners[:, 0, 1], 'ro')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        
        axs[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Re-projected corners')
        axs[1].plot(imgpoints2[:, 0, 0], imgpoints2[:, 0, 1], 'bo')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        plt.show()

# This code is used to find the normal vector to the checkerboard's plane in the camera frame
# for each image in the list of images.

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    
    if ret == True:
        
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [corners], gray.shape[::-1])
        R = cv2.Rodrigues(rvecs[0])[0]
        R = np.linalg.inv(R)
        n_c = np.dot(R, [0, 0, 1])
        
        print(f'Image {i+1}: n^c = {n_c}')
        # display(Latex(f'Image {i+1}: $n^c$ = {n_c}'))