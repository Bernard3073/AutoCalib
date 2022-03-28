import cv2
import glob
import numpy as np
# from scipy.optimize import

# Defining the dimensions of checkerboard
CHECKERBOARD_SIZE = 21.5  # inch
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Creating vector to store vectors of 3D points for each checkerboard image
objpoints = []
# Creating vector to store vectors of 2D points for each checkerboard image
imgpoints = []
# Defining the world coordinates
# world_coord = np.array([[CHECKERBOARD_SIZE, CHECKERBOARD_SIZE],
#                          [CHECKERBOARD_SIZE * CHECKERBOARD[0], CHECKERBOARD_SIZE],
#                          [CHECKERBOARD_SIZE * CHECKERBOARD[0], CHECKERBOARD_SIZE * CHECKERBOARD[1]],
#                          [CHECKERBOARD_SIZE, CHECKERBOARD_SIZE*CHECKERBOARD[1]]], np.float32)
homographies = []


def est_homography(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    world_coord_x, world_coord_y = np.meshgrid(
        range(CHECKERBOARD[0]), range(CHECKERBOARD[1]))
    world_coord = np.hstack((world_coord_x.reshape(
        54, 1), world_coord_y.reshape(54, 1))).astype(np.float32)
    world_coord *= CHECKERBOARD_SIZE
    world_coord = np.array(world_coord)
    if ret == True:
        # objpoints.append(world_coord)
        # reshape corners from (54, 1, 2) to (54, 2)
        corners = corners.reshape(corners.shape[0], 2)
        corners = corners.reshape(-1, 2)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria)
        # imgpoints.append(corners2)
        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        H, _ = cv2.findHomography(world_coord, corners2)
        homographies.append(H)
        # cv2.imshow('img', img)
        # cv2.waitKey(300)
    else:
        print('No checkerboard found')
        return None
    # cv2.destroyAllWindows()
    return homographies


def v_mat(H, i, j):
    v = np.array([[H[0][i] * H[0][j]],
                  [H[0][i] * H[1][j] + H[0][j] * H[1][i]],
                  [H[1][i] * H[1][j]],
                  [H[2][i] * H[0][j] + H[0][i] * H[2][j]],
                  [H[2][i] * H[1][j] + H[1][i] * H[2][j]],
                  [H[2][i] * H[2][j]]])

    return np.reshape(v, (6, 1))
    # return v.T


def v(p, q, H):
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])

def getVij(hi, hj):
    Vij = np.array([ hi[0]*hj[0], hi[0]*hj[1] + hi[1]*hj[0], hi[1]*hj[1], hi[2]*hj[0] + hi[0]*hj[2], hi[2]*hj[1] + hi[1]*hj[2], hi[2]*hj[2] ])
    return Vij.T

def getV(all_H):
    v = []
    for H in all_H:
        h1 = H[:,0]
        h2 = H[:,1]

        v12 = getVij(h1, h2)
        v11 = getVij(h1, h1)
        v22 = getVij(h2, h2)
        v.append(v12.T)
        v.append((v11 - v22).T)
    return np.array(v)

def est_intrinsic_param(homographies):
    # V = []
    # for h in homographies:
    #     # V.append(v_mat(h, 0, 1).T)
    #     # V.append((v_mat(h, 0, 0) - v_mat(h, 1, 1)).T)
    #     V.append(v(0, 1, h))
    #     V.append(v(0, 0, h) - v(1, 1, h))

    # V = np.array(V)
    V = getV(homographies)
    _, S, Vt = np.linalg.svd(V)
    b = Vt[-1, :]

    B11 = b[0]
    B12 = b[1]
    B22 = b[2]
    B13 = b[3]
    B23 = b[4]
    B33 = b[5]

    v0 = (B12*B13 - B11*B23)/(B11*B22 - B12**2)
    lamda = B33 - (B13**2 + v0*(B12*B13 - B11*B23))/B11

    alpha = np.sqrt(lamda/B11)
    beta = np.sqrt(lamda*B11 / (B11*B22 - B12**2))
    gamma = -1*B12*(alpha**2)*beta/lamda
    u0 = gamma*v0/beta - B13*(alpha**2)/lamda
    # camera intrinic matrix
    A_mat = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
    return A_mat


def est_extrinsic_param(A):
    A_inv = np.linalg.inv(A)
    # Rotation vectors
    r1 = np.dot(A_inv, A[:, 0])
    lamda = np.linalg.norm(r1, ord=2)
    r1 = r1/lamda
    r2 = np.dot(A_inv, A[:, 1])/lamda
    r3 = np.cross(r1, r2)
    # Translation vectors
    t = np.dot(A_inv, A[:, 2])/lamda
    R = np.array([r1, r2, r3])
    R = R.T

    return R, t


def main():
    images = [cv2.imread(file)
              for file in glob.glob('./Calibration_Imgs/*.jpg')]
    for img in images:
        homographies = est_homography(img)
        A_mat = est_intrinsic_param(homographies)
        R, t = est_extrinsic_param(A_mat)
        print(R, t)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
