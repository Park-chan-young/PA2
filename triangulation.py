import numpy as np

def triangulate_points(EM0, EM1, inlier_p1, inlier_p2, camera_intrinsic):
    """
    Step4: Computes 3D points via linear triangulation using the given camera poses (EM0, EM1)
    and the corresponding inlier keypoint coordinates from two images.
    
    Allow functions:
        numpy
        
    Deny functions:
        cv2

    Parameters:
        EM0             : Pose of the first camera ([I|0], 3x4 numpy array).
        EM1             : Pose of the second camera (3x4 numpy array).
        inlier_p1       : Inlier keypoints from the first image (N x 2 numpy array, [x, y]).
        inlier_p2       : Inlier keypoints from the second image (N x 2 numpy array, [x, y]).
        camera_intrinsic: Camera intrinsic matrix (3x3 numpy array).
        
    Returns:
        points_3d (np.ndarray): (N x 3) numpy array where each row is the triangulated 3D coordinate (X, Y, Z).
        inlier_idx (np.ndarray): (N,) numpy array containing the indices of the inlier points used.
    """
    #TODO: Fill this functions
    
    # 이전 decompostion 모듈에서 triangulation 한 부분 사용
    P0 = EM0
    P1 = EM1
    points_3d = []
    inlier_idx = []
    for i in range(len(inlier_p1)):
        x0, y0 = inlier_p1[i]
        x1, y1 = inlier_p2[i]
        
        A = [x0*P0[2] - P0[0],
                y0*P0[2] - P0[1],
                x1*P1[2] - P1[0],
                y1*P1[2] - P1[1]]
        A = np.array(A)
        _U, _S, _Vt = np.linalg.svd(A)
        X = _Vt[-1]
        #print(X) #(x,y,z,w)
        #break
        _w = X[3]
        X = X/_w #3D point
        points_3d.append(X[:3]) #주석 (N 3)
        inlier_idx.append(i)
        
    points_3d = np.array(points_3d)
    inlier_idx = np.array(inlier_idx)
            

    
    
    return points_3d, inlier_idx
