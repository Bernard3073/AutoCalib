import argparse
import cv2
import glob
import numpy as np
import scipy.optimize as opt

# Defining the dimensions of checkerboard
CHECKERBOARD_SIZE = 21.5  # inch
CHECKERBOARD = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def est_homography(img_list, world_pts, save_folder):
    homographies = []
    corner_pts_list = []
    for i, img in enumerate(img_list):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if ret == True:
            # reshape corners from (54, 1, 2) to (54, 2)
            corners = corners.reshape(corners.shape[0], 2)
            corners = corners.reshape(-1, 2)
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            H, _ = cv2.findHomography(world_pts, corners2, cv2.RANSAC, 5.0)
            homographies.append(H)
            corner_pts_list.append(corners2)
            # cv2.imshow('img', img)
            # cv2.imwrite(save_folder + '/' + str(i) + '_corners.png', img)
            # cv2.waitKey(0)
        else:
            print('No checkerboard found')
            return None
    return homographies, corner_pts_list

def v_mat(p, q, H):
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1, p] * H[0, q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])

def est_intrinsic_param(homographies):
    V = []
    for h in homographies:
        V.append(v_mat(0, 1, h))
        V.append(v_mat(0, 0, h) - v_mat(1, 1, h))

    V = np.array(V)
    _, _, Vt = np.linalg.svd(V)
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
    u0 = (gamma*v0/beta) - (B13*(alpha**2)/lamda)
    # camera intrinic matrix
    A_mat = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [0, 0, 1]])
    return A_mat


def est_extrinsic_param(A, homographies):
    A_inv = np.linalg.inv(A)
    rot_tran_list = []
    for H in homographies:
        # Rotation vectors
        r1 = np.dot(A_inv, H[:, 0])
        lamda = np.linalg.norm(r1, ord=2)
        r1 = r1/lamda
        r2 = np.dot(A_inv, H[:, 1])/lamda
        r3 = np.cross(r1, r2)
        # Translation vectors
        t = np.dot(A_inv, H[:, 2])/lamda
        R = np.array([r1, r2, r3])
        R = R.T
        # extrinsic = np.zeros((3, 4))
        # extrinsic[:, :-1] = R
        # extrinsic[:, -1] = t
        rot_tran = np.vstack((r1, r2, r3, t)).T
        rot_tran_list.append(rot_tran)

    return rot_tran_list

def loss_func(params, corner_pts_list, world_pts, rot_tran_list):
    alpha, gamma, beta, u0, v0, k1, k2 = params
    A_mat = np.array([[alpha, gamma, u0],
                      [0, beta, v0],
                      [k1, k2, 1]])
    error_mat = []
    for i, corner in enumerate(corner_pts_list):
        rot_tran = rot_tran_list[i]

        r1r2t = np.array((rot_tran[:, 0], rot_tran[:, 1], rot_tran[:, 3])).reshape(3, 3)
        r1r2t_t = r1r2t.T
        pred_rt = np.dot(A_mat, r1r2t_t)
        error_sum = 0
        for j in range(len(world_pts)):
            world_pts_2d = world_pts[j]
            world_pts_2d_homo = np.array([world_pts_2d[0], world_pts_2d[1], 1]).reshape(3, 1)
            world_pts_3d_homo = np.array([world_pts_2d[0], world_pts_2d[1], 0, 1]).reshape(4, 1)
            proj_pts = np.dot(rot_tran, world_pts_3d_homo)
            # proj_pts = np.matmul(rot_tran, world_pts_3d_homo)
            proj_pts /= proj_pts[2]
            x, y = proj_pts[0], proj_pts[1]
            U = np.dot(pred_rt, world_pts_2d_homo)
            U = U/U[2]
            u, v = U[0], U[1]
            mij = corner[j]
            mij = np.array([mij[0], mij[1], 1], dtype = 'float32').reshape(3,1)
            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            mij_cap = np.array([u_cap, v_cap, 1], dtype = 'float32').reshape(3,1)
            error = np.linalg.norm((mij - mij_cap), ord=2)
            error_sum += error
        error_mat.append(error_sum / 54)
    
    return np.array(error_mat)

def optimization(A, corner_pts_list, world_pts, rot_tran_list):
    alpha = A[0, 0]
    gamma = A[0, 1]
    u0 = A[0, 2]
    beta = A[1, 1]
    v0 = A[1, 2]
    k1 = A[2, 0]
    k2 = A[2, 1]
    optimize_params = opt.least_squares(fun=loss_func, x0 = [alpha, gamma, beta, u0, v0, k1, k2], 
        method = 'lm', args=(corner_pts_list, world_pts, rot_tran_list))
    [alpha_opt, gamma_opt, beta_opt, u0_opt, v0_opt, k1_opt, k2_opt] = optimize_params.x
    A_opt = np.array([[alpha_opt, gamma_opt, u0_opt],
                      [0, beta_opt, v0_opt],
                      [0, 0, 1]])
    return A_opt, k1_opt, k2_opt

def reproject_error(A, kc, rot_tran_list, corner_pts_list, world_pts):
    u0 = A[0, 2]
    v0 = A[1, 2]
    k1, k2= kc[0], kc[1]

    error_mat = []
    reproj_pts_list = []
    for i, corner in enumerate(corner_pts_list):
        rot_tran = rot_tran_list[i]

        r1r2t = np.array((rot_tran[:, 0], rot_tran[:, 1], rot_tran[:, 3])).reshape(3, 3)
        r1r2t_t = r1r2t.T
        pred_rt = np.dot(A, r1r2t_t)
        error_sum = 0
        reproj_pts = []
        for j in range(world_pts.shape[0]):
            world_pts_2d = world_pts[j]
            world_pts_2d_homo = np.array([world_pts_2d[0], world_pts_2d[1], 1]).reshape(3, 1)
            world_pts_3d_homo = np.array([world_pts_2d[0], world_pts_2d[1], 0, 1]).reshape(4, 1)
            proj_pts = np.dot(rot_tran, world_pts_3d_homo)
            # proj_pts = np.matmul(rot_tran, world_pts_3d_homo)
            proj_pts /= proj_pts[2]
            x, y = proj_pts[0], proj_pts[1]
            U = np.dot(pred_rt, world_pts_2d_homo)
            U /= U[2]
            u, v = U[0], U[1]
            mij = corner[j]
            mij = np.array([mij[0], mij[1], 1], dtype = 'float32').reshape(3,1)
            t = x**2 + y**2
            u_cap = u + (u-u0)*(k1*t + k2*(t**2))
            v_cap = v + (v-v0)*(k1*t + k2*(t**2))
            reproj_pts.append([u_cap, v_cap])
            mij_cap = np.array([u_cap, v_cap, 1], dtype = 'float32').reshape(3,1)
            error = np.linalg.norm((mij - mij_cap), ord=2)
            error_sum += error
        error_mat.append(error_sum)
        reproj_pts_list.append(reproj_pts)
    error_mat = np.array(error_mat)
    error_avg = np.sum(error_mat) / (len(corner_pts_list) * world_pts.shape[0])
    return error_avg, reproj_pts_list

def main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--data_folder', default= './Calibration_Imgs/', help='directory of images')
    Parser.add_argument('--output_folder', default='./output/', help='directory to save the output')

    Args = Parser.parse_args()
    data_folder = Args.data_folder
    save_folder = Args.output_folder 
    images = [cv2.imread(file)
              for file in glob.glob(data_folder + '*.jpg')]
    world_pts_x, world_pts_y = np.meshgrid(
            range(CHECKERBOARD[0]), range(CHECKERBOARD[1]))
    world_pts = np.hstack((world_pts_x.reshape(
        54, 1), world_pts_y.reshape(54, 1))).astype(np.float32)
    world_pts *= CHECKERBOARD_SIZE
    world_pts = np.array(world_pts)
    homographies, corner_pts_list = est_homography(images, world_pts, save_folder)
    A_mat = est_intrinsic_param(homographies)
    rot_tran_list = est_extrinsic_param(A_mat, homographies)
    A_mat_opt, k1, k2 = optimization(A_mat, corner_pts_list, world_pts, rot_tran_list)

    kc = np.array([0, 0]).reshape(2, 1)
    kc_opt = np.array([k1, k2]).reshape(2, 1)
    new_rot_tran_list = est_extrinsic_param(A_mat_opt, homographies)
    reproj_error, reproj_pts_list = reproject_error(A_mat_opt, kc_opt, new_rot_tran_list, corner_pts_list, world_pts)
    
    K = np.array(A_mat_opt, np.float32).reshape(3,3)
    D = np.array([kc_opt[0],kc_opt[1], 0, 0] , np.float32)
    print("K matrix: \n", K)
    print('optimized kc: ', kc_opt)
    print("re-project error: ", reproj_error)
    for i,image_points in enumerate(reproj_pts_list):
        image = cv2.undistort(images[i], K, D)
        for point in image_points:
            x = int(point[0])
            y = int(point[1])
            image = cv2.circle(image, (x, y), 5, (0, 0, 255), 10)
        # cv2.imshow('frame', image)
        cv2.imwrite(save_folder + "rectify_" + str(i) + ".png", image)
        # cv2.waitKey(0)

    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
