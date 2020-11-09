"""
Image Stitching Problem
(Due date: Nov. 9, 11:59 P.M., 2020)
The goal of this task is to stitch two images of overlap into one image.
To this end, you need to find feature points of interest in one image, and then find
the corresponding ones in another image. After this, you can simply stitch the two images
by aligning the matched feature points.
For simplicity, the input two images are only clipped along the horizontal direction, which
means you only need to find the corresponding features in the same rows to achieve image stiching.

Do NOT modify the code provided to you.
You are allowed use APIs provided by numpy and opencv, except “cv2.findHomography()” and
APIs that have “stitch”, “Stitch”, “match” or “Match” in their names, e.g., “cv2.BFMatcher()” and
“cv2.Stitcher.create()”.
"""
import cv2
import numpy as np
import random

def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: you need to return the result image which is stitched by left_img and right_img
    """

    #sift initialization - Scale Invariant features
    sift   = cv2.SIFT_create()
    
    #Left Image Key Points
    leftimg= cv2.cvtColor(left_img,cv2.COLOR_BGR2GRAY)
    leftkeypnts= sift.detect(leftimg,None)
    
    #Right Image Key Points
    rightimg= cv2.cvtColor(right_img,cv2.COLOR_BGR2GRAY)
    rightkeypnts= sift.detect(rightimg,None)
    
    #Extract key point features descriptors for both left and right image
    #reducing search area of match points in left and right images
    leftsrcharea=len(leftkeypnts)//2     
    rightsrcharea=len(rightkeypnts)//2   
    
    leftkeypnts,leftfkdes=sift.compute(leftimg,leftkeypnts[leftsrcharea:])
    rightkeypnts,rightfkdes=sift.compute(rightimg,rightkeypnts[:rightsrcharea])
    
    #######################################Coded Matching Algorithm######################################## 
    ratio=0.85  #Lowe's Ratio Test to identify good matching points
    kNN=2
    keymatches=[]
    
    #Calculating euclidean distance(l2 norm) from each point in left to all points in right image
    #Find best matching point having lowest distance
    
    dist, nidx=cv2.batchDistance(leftfkdes, rightfkdes, K=kNN, dtype=-1, normType = cv2.NORM_L2)
    

    for ld in range(len(leftkeypnts)):   
        matchperpt=[]
        for i in range(kNN):
            matchperpt.append([ld,nidx[ld,i],dist[ld,i]])
            
        keymatches.append(matchperpt)
        
    best_match = []

    for m1, m2 in keymatches:
        if m1[2] < ratio * m2[2]:
            best_match.append((m1[0], m1[1]))

    #####################################################################################################
    
    ###################################Finding Homography Matrix#########################################
    #To find homography matrix we need 4 good matching points
    
    MIN_MATCH_COUNT = 4
    RANSAC_THRESH=5.0

    if len(best_match) > MIN_MATCH_COUNT:
        src_pts = np.float32([ leftkeypnts[m[0]].pt for m in best_match ]).reshape(-1,1,2)
        dst_pts = np.float32([ rightkeypnts[m[1]].pt for m in best_match ]).reshape(-1,1,2)
        
        H, mask = cv2.findHomography(dst_pts,src_pts, cv2.RANSAC, RANSAC_THRESH)
    
    ###################################################################################################
    #Wrap Perspective - to identify the position wrt to overlap part
    #Stitch by joining the images handling the overlap part using wrap perspective
    #eliminate excess points
    
        left_height= left_img.shape[0]
        left_width = left_img.shape[1]
        right_width= right_img.shape[1]
        new_height = left_height
        new_width  = left_width + right_width
    
        result = cv2.warpPerspective(right_img, H, (new_width, new_height))
        result[0:left_height,0:left_width]=left_img
        
        #Eliminating the excess ends
        for i in range(new_width-1,0,-1):
            if len(np.unique(result[:,i]))==1 and np.unique(result[:,i])==0:
                result=np.delete(result,i,axis=1)
            else:
                break
                
        
        return result
  
    ####################################################################################################
    

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_image = solution(left_img, right_img)
    cv2.imwrite('results/task1_result.jpg',result_image)