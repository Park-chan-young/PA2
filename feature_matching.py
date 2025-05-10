import cv2
import glob
import numpy as np
import os
from tqdm import tqdm

def matching_two_image(image1_path, image2_path, threshold_knn=0.75):
    """
    TODO:
    Step1: Accepts two image file paths and performs SIFT-based matching between them.
    It detects keypoints and computes descriptors using the SIFT algorithm,
    then matches descriptors using BFMatcher with k-NN matching.
    Finally, it applies Lowe's ratio test to filter out unreliable matches.
    
    Allow functions:
        cv2.cvtColor()
        cv2.SIFT_create()
        cv2.SIFT_create().*
        cv2.BFMatcher()
        cv2.BFMatcher().*
        cv2.drawMatchesKnn()
        
    Parameters:
        image1_path (str): File path for the first image.
        image2_path (str): File path for the second image.
        threshold_knn (float): Lowe's ratio test threshold (default is 0.75).
        
    Output:
        img1, img2 (numpy.ndarray): The original images.
        kp1, kp2 (list[cv2.KeyPoint]): Lists of keypoints detected in each image.
        des1, des2 (numpy.ndarray): SIFT descriptors for each image.
        matches (list[cv2.DMatch]): The matching results after applying Lowe's ratio test.
    """
    #TODO: Fill this functions
    
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    #kp1, kp2, des1, des2 구하기:
    sift = cv2.SIFT_create()
    kp1 = sift.detect(img1)
    kp2 = sift.detect(img2)
    kp1, des1 = sift.compute(img1, kp1)
    kp2, des2 = sift.compute(img2, kp2)
    
    # matches 구하기
    matches = []
    bf = cv2.BFMatcher()
    kmatch = bf.knnMatch(des1, des2, k=2) #des1의 각 descripter에 대해, 최대 k개 매칭 ... Lowe,s ratio 적용 k= 2
    
    # example check
    # f, s = kmatch[0]
    # print("first: %d, second: %d" % (f.distance, s.distance))
    
    if len(kmatch) == 0:
        print("check no found!")
        matches = []
    
    for its in kmatch:
        
        if(len(its) < 2):
            print("check1")
            continue
        f, s = its
        if f.distance < threshold_knn*s.distance:
            matches.append(f)
        
    
    
    return img1, img2, kp1, kp2, des1, des2, matches
