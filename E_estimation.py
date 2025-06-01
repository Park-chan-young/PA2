import numpy as np
from tqdm import tqdm
import matlab.engine

def essential_matrix_estimation(kp1, kp2, matches, camera_intrinsic, eng, 
                                max_iter=5000, threshold=1e-5):
    """
    Step2: Estimates the Essential Matrix using the 5-Point Algorithm with RANSAC.
    It takes the camera intrinsic matrix, keypoints from two images, their matches,
    and a MATLAB engine instance (which must already be started). The function uses
    a RANSAC loop to find the best Essential Matrix candidate that fits the normalized
    matched keypoints.

    Allow functions:
        numpy
        tqdm (for progress tracking)
        eng.calibrated_fivepoint() (please read ./Step2/calibrated_fivepoint.m)

    Deny functions:
        cv2

    Parameters:
        kp1 (list): List of cv2.KeyPoint objects from the first image.
        kp2 (list): List of cv2.KeyPoint objects from the second image.
        matches (list): List of cv2.DMatch objects representing the matches between the images.
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3).
        eng: MATLAB engine object (already started).
        max_iter (int): Maximum number of RANSAC iterations (default 5000).
        threshold (float): Inlier threshold for error (default 1e-5).

    Returns:
        E_est (np.ndarray): The estimated Essential Matrix (3x3).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2).
        best_inlier_idx (np.ndarray): Inlier matching index (N, )
    """
    # TODO: Fill this function
    
    #초기값 설정
    max_inliers = 0
    best_E = None
    best_inlier_idx = []
    
    #cv2 to numpy
    
    # ex_kp1 = kp1[0]
    # print(dir(ex_kp1))
    # print(ex_kp1.pt)   
    # print(type(ex_kp1.pt)) #kp1.pt: 특징점 좌표로 나옴... class tuple
    
    kp1_np = np.array([kp_1.pt for kp_1 in kp1])
    kp2_np = np.array([kp_2.pt for kp_2 in kp2])
    
    # ex_matches = matches[0]
    # print(dir(ex_matches))
    
    matches_list = []
    
    for match in matches:
        kp1_idx = match.queryIdx
        kp2_idx = match.trainIdx
        
        kp1_x, kp1_y = kp1_np[kp1_idx]
        kp2_x, kp2_y = kp2_np[kp2_idx]
        
        matches_list.append([kp1_x, kp1_y, kp2_x, kp2_y])

    matches_np = np.array(matches_list)
    p1 = matches_np[:, :2]
    p2 = matches_np[:, 2:]
    
    #homogeneous 행태로 변환 by np.concate
    N = p1.shape[0]
    ones_col = np.ones((N,1))
    p1_h = np.concatenate((p1, ones_col), axis = 1)
    p2_h = np.concatenate((p2, ones_col), axis = 1)
    
    #정규화
    camera_intrinsic_inv = np.linalg.inv(camera_intrinsic)
    p1_norm = (camera_intrinsic_inv @ p1_h.T).T
    p2_norm = (camera_intrinsic_inv @ p2_h.T).T
    
    #RANSAC by matlab
    
    for _ in range(max_iter):
        rand_idx = np.random.choice(len(p1_norm), 5, replace = False)
        p1_sample = p1_norm[rand_idx]
        p2_sample = p2_norm[rand_idx]
        
        #for matlab 
        p1_forMAT = p1_sample.T.tolist()
        P2_forMAT = p2_sample.T.tolist()
        
        p1_mat = matlab.double(p1_forMAT)
        p2_mat= matlab.double(P2_forMAT)
        E_cand = eng.calibrated_fivepoint(p1_mat, p2_mat)
        E_cand = np.array(E_cand)
        # print(E_cand.shape) # (9, N)        
        for i in range(E_cand.shape[1]):
            E = E_cand[:, i].reshape(3, 3)
            
            errors = []
            for i in range(p1_norm.shape[0]):
                x1 = p1_norm[i].reshape(1, 3)
                x2 = p2_norm[i].reshape(1, 3)
                err = np.abs(x2 @ E @ x1.T)[0][0]
                errors.append(err)
            
            current_inliers = []
            for i in range(len(errors)):
                if errors[i] < threshold:
                    current_inliers.append(i)
            
            if len(current_inliers) > max_inliers:
                max_inliers = len(current_inliers)
                best_E = E
                best_inlier_idx = current_inliers
        
        E_est = best_E
        inlier_p1 = p1[best_inlier_idx]
        inlier_p2 = p2[best_inlier_idx]


    
    return E_est, inlier_p1, inlier_p2, best_inlier_idx