import numpy as np
import convolution
import timeit
from skimage import measure,morphology
import shutil
import os
import cv2
import copy
import sys
from kmean_iteration import connected, initial, check_d, find_R, renew_params
from kmean_merges import merge_small, calc_new_ycbcr, merge 
# from kmean_func import initial, check_d, find_R, renew_params, merge, output1, output2 

# how many k means 
k_mean = 62
# order of color :[B,G,R]
img2 = cv2.imread('hans.bmp')
# how many iterations
times = 6

def k_means(img,times):
    # M, N = height, width
    # d(m,n,k) =  the evaluation value of every pixel to each mean(kmeans)
    # higher d means higher correlation
    # ycbcr = corresponding y,cb,cr to R,G,B of each mean(kmeans)
    global M,N,k_mean,d,R,ycbcr
    global sobel_x
    global grad_nor

    size = img.shape
    M,N = size[0],size[1]
    
    # downsample (if M>600 and N>600)    
    if M > 600 and N > 300:
        if M > N:
            img = cv2.resize(img, None, fx=1/int(M/300), fy=1/int(M/300), interpolation = cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, None, fx=1/int(N/300), fy=1/int(N/300), interpolation = cv2.INTER_CUBIC)
        size = img.shape
        M,N = size[0],size[1]
    elif M > 300 and N > 600:
        if N > M:
            img = cv2.resize(img, None, fx=1/int(N/300), fy=1/int(N/300), interpolation = cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, None, fx=1/int(M/300), fy=1/int(M/300), interpolation = cv2.INTER_CUBIC)
        size = img.shape
        M,N = size[0],size[1]
    
    # sober filter
    sobel_x = np.array(([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),dtype=int)
    sobel_y = sobel_x.T
    new_image_x = convolution.convolution(img, sobel_x)
    new_image_y = convolution.convolution(img, sobel_y)
    grad = np.sqrt(np.square(new_image_x) + np.square(new_image_y))
    # print(grad.max())
    grad_nor =  grad * 255.0 / grad.max()
    print(grad_nor[0,0])
    start = timeit.default_timer()
   
    # initialization phase
    new_coord, mean_set = initial(img,size,k_mean,grad_nor)
    end = timeit.default_timer()
    print(f'time for initialization: {end - start}')
    ycbcr = np.zeros([k_mean,3])
    # store y,cb,cr of means
    for k in range(k_mean):
        ycbcr[k] = new_coord[int(mean_set[k][0])][int(mean_set[k][1])]
    
    # iteratively renew parameters
    R = np.zeros([M,N],dtype=int)
    d = np.ones([M,N,k_mean])
    d = d*10000 # big number (because we use argmin() to find R) 
    for time in range(times):
        # evaluation d(correlation of mean & pixel)
        start = timeit.default_timer()
        check_d(new_coord, mean_set, size, k_mean, d, ycbcr)
        end = timeit.default_timer()
        print(f'time for generate d: {end - start}')

        # assign fittest region to every pixel
        start = timeit.default_timer()
        find_R(size, d, R)
        end = timeit.default_timer()
        print(f'time for generate R: {end - start}')

        # evaluate new parameters
        start = timeit.default_timer()
        print(f"cycle {time} to renew!!")
        random_params = [grad_nor,size]
        mean_set = renew_params(new_coord, mean_set, k_mean, ycbcr, R, random_params)
        end = timeit.default_timer()
        print(f'time for renew params: {end - start}')
    
    ### merge phase ###
    # divide inconnected regions and merge small regions
    r_k = np.unique(R)
    min_region_area = 30
    conn_R = connected(r_k,R,size)
    nosmall_R = morphology.remove_small_objects(conn_R,min_size=min_region_area,connectivity=2)
    mergesmall_R = merge_small(nosmall_R, size, ycbcr)
    
    # merge adjacent regions according to params
    remain_kmean = np.unique(mergesmall_R)
    new_ycbcr = calc_new_ycbcr(new_coord,remain_kmean,mergesmall_R,ycbcr)
    merge_R = merge(remain_kmean,mergesmall_R,new_ycbcr,img, grad_nor, k_mean, min_region_area)
    final_kmean = np.unique(merge_R)
    
    output(img,merge_R,final_kmean)
    
def output(img,merge_R,final_kmean):
    global k_mean
    print('following are output with measure.label')
    #print(f"R: {merge_R[:15,:15]}")
    parent_dir = 'week14'
    dir_name = f'{k_mean}means'
    path = os.path.join(parent_dir, dir_name)
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
    os.mkdir(path)
    # max = np.amax(merge_R)
    # print(f'max: {max}')
    region = 0
    color_R = 0
    color_G = 0
    color_B = 0
    img_out = copy.copy(img)
    for region in final_kmean:
        r1_i, r1_j = np.where(merge_R == region)
        if len(r1_i) != 0:
            for p in range(len(r1_i)):
                bgr = np.array([color_B%256,color_G%256,color_R%256])
                img_out[r1_i[p]][r1_j[p]] = bgr
            color_R += 8
            color_G += 24
            color_B += 40
        else:
            print(f'region {region} is empty')
    cv2.imwrite(f'week14/{k_mean}means/output{region}.jpg', img_out)
        

k_means(img2,times)