import numpy as np

def essential_matrix_decomposition(E, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step3: Decomposes the Essential Matrix and performs triangulation to compute the poses of the two cameras.
    The function returns the pose of the first camera (P0, which is [I | 0]) and the selected pose for
    the second camera (P1) based on the cheirality condition.

    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        E (np.ndarray): Essential Matrix (3x3 numpy array).
        inlier_p1 (np.ndarray): Inlier keypoint coordinates from the first image (N x 2 numpy array).
        inlier_p2 (np.ndarray): Inlier keypoint coordinates from the second image (N x 2 numpy array).
        camera_intrinsic (np.ndarray): Camera intrinsic matrix (3x3 numpy array).

    Returns:
        P0 (np.ndarray): Pose of the first camera ([I | 0], 3x4 numpy array).
        P1 (np.ndarray): Pose of the selected second camera (3x4 numpy array).
    """
    #TODO: Fill this functions
    
    # 기준 카메라 P0
    I = [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]]
    I = np.array(I)
    R_0 = [[0],
           [0],
           [0]]
    R_0 = np.array(R_0)
    P0 = np.concatenate((I, R_0), axis = 1)
    
    #SVD 분해
    U, S, Vt = np.linalg.svd(E)
    
    W = [[0, -1, 0],
         [1,  0, 0],
         [0,  0, 1]]
    W = np.array(W)
    

    
    #4 가지 후보 생성
    R1_a = U @ W @ Vt
    R1_b = U @ W.T @ Vt
    u3 = U[:, 2]
    re_u3 = u3.reshape(-1, 1) #concaat 할떄 shape 주의하자!!!!
    P1_cand = []
    P1_cand.append(np.concatenate((R1_a, re_u3), axis = 1))
    P1_cand.append(np.concatenate((R1_a, -re_u3), axis = 1))
    P1_cand.append(np.concatenate((R1_b, re_u3), axis = 1))
    P1_cand.append(np.concatenate((R1_b, -re_u3), axis =1))
    
    # cheirality condition: 4가지 중 하나 선택하기(Z > 0)
    P1 = None
    Mcount = -1
    
    for temp_p1 in P1_cand:
        count = 0
        for i in range(len(inlier_p1)):
            x0, y0 = inlier_p1[i]
            x1, y1 = inlier_p2[i]
            
            A = [x0*P0[2] - P0[0],
                 y0*P0[2] - P0[1],
                 x1*temp_p1[2] - temp_p1[0],
                 y1*temp_p1[2] - temp_p1[1]]
            A = np.array(A)
            _U, _S, _Vt = np.linalg.svd(A)
            X = _Vt[-1]
            #print(X) #(x,y,z,w)
            #break
            _w = X[3]
            X = X/_w #3D point
            chk_X = X.reshape(4, 1) #주의할점: 여기서 많이 해맸는데, 이거 까먹지 말자 @ 때문에 shape 맞춰주기
            X_img0 = P0 @ chk_X
            X_img1 = temp_p1 @ chk_X
            
            if X_img0[2] > 0 and X_img1[2] > 0:
                count = count+1
            
        if count > Mcount:
            P1 = temp_p1
            Mcount = count    

    
    return P0, P1
