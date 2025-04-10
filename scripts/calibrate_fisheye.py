import cv2
import numpy as np
import glob

# Fisheye calibration parameters
CHECKERBOARD = (6,9)  # Inner corners pattern
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Arrays to store object/image points
objpoints = []  # 3D world points
imgpoints = []  # 2D image points

images = glob.glob('calibration_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(
        gray, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (3,3), (-1,-1), criteria)
        imgpoints.append(corners2)

# Fisheye calibration
K = np.zeros((3,3))
D = np.zeros((4,1))
rvecs = [np.zeros((1,1,3), dtype=np.float64) for _ in imgpoints]
tvecs = [np.zeros((1,1,3), dtype=np.float64) for _ in imgpoints]

ret, K, D, _, _ = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    K,
    D,
    rvecs,
    tvecs,
    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
)

print(f"Camera Matrix:\n{K}")
print(f"Distortion Coefficients:\n{D}")