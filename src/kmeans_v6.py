from site import addpackage
from cv2 import Laplacian
import numpy as np
import cv2
import convolution
import timeit
from skimage import measure,morphology
import shutil
import os
from cv2 import sqrt
import cv2
import numpy as np
import random
import math
import copy
import sys
# from kmean_func import initial, check_d, find_R, renew_params, merge, output1, output2 

# how many k means 
k_mean = 80
# order of color :[B,G,R]
img2 = cv2.imread('019.BMP')
# how many iterations
times = 4

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
        new_size = img.shape
        M,N = new_size[0],new_size[1]
    elif M > 300 and N > 600:
        if N > M:
            img = cv2.resize(img, None, fx=1/int(N/300), fy=1/int(N/300), interpolation = cv2.INTER_CUBIC)
        else:
            img = cv2.resize(img, None, fx=1/int(M/300), fy=1/int(M/300), interpolation = cv2.INTER_CUBIC)
        new_size = img.shape
        M,N = new_size[0],new_size[1]
    
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
    new_coord, mean_set = initial(img)
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
        check_d(new_coord,mean_set)
        end = timeit.default_timer()
        print(f'time for generate d: {end - start}')

        # assign fittest region to every pixel
        start = timeit.default_timer()
        find_R()
        end = timeit.default_timer()
        print(f'time for generate R: {end - start}')

        # evaluate new parameters
        start = timeit.default_timer()
        print(f"cycle {time} to renew!!")
        mean_set,ept_set,B = renew_params(new_coord,mean_set)
        end = timeit.default_timer()
        print(f'time for renew params: {end - start}')
    
    # output1(img,R,0)
    #conn_R = measure.label(R,connectivity=2)  #8连通区域标记
    r_k = np.unique(R)
    conn_R = connected(r_k,R)
    # output1(img,conn_R,1)
    nosmall_R = morphology.remove_small_objects(conn_R,min_size=30,connectivity=2)
    mergesmall_R = merge_small(nosmall_R)
    
    # print(f"things about R(merge): size: {mergesmall_R.shape} \nR: {mergesmall_R}")
    remain_kmean = np.unique(mergesmall_R)
    new_ycbcr = calc_new_ycbcr(new_coord,remain_kmean,mergesmall_R)
    merge_R = merge(remain_kmean,mergesmall_R,new_ycbcr,img)
    final_kmean = np.unique(merge_R)
    output2(img,merge_R,final_kmean)
    #print(mean_set)

def connected(remain_kmean,merge_R):
    global M,N 
    connected_R = np.zeros([M+2,N+2],int)
    section = 1
    for region in remain_kmean:
        region_pad = np.zeros([M+2,N+2],int)
        region_pad[1:M+1,1:N+1] = np.isin(merge_R,region).astype(int)
        for i in range(1,M+1):
            for j in range(1,N+1):
                if region_pad[i,j] == 1:
                    if region_pad[i-1,j] == 0 and region_pad[i,j-1] == 0:
                        connected_R[i,j] = section
                        section += 1
                    elif region_pad[i-1,j] == 1 and region_pad[i,j-1] == 0:
                        connected_R[i,j] = connected_R[i-1,j]
                    elif region_pad[i-1,j] == 0 and region_pad[i,j-1] == 1:
                        connected_R[i,j] = connected_R[i,j-1]
                    else:
                        s_up = connected_R[i-1,j]
                        s_left = connected_R[i,j-1]
                        connected_R[i,j] = s_up
                        if s_up != s_left:
                            diff = int(s_left - s_up)
                            connected_R =  connected_R - np.isin(connected_R,s_left).astype(int)*diff
                            #print(connected_R)
    return connected_R[1:M+1,1:N+1]


def initial(img):
    global M,N,k_mean,grad_nor
    L = 10
    coord = np.array([[0.114,0.587,0.299],[0.5,-0.331,-0.169],[-0.081,-0.419,0.5]])
    new_coord = np.zeros([M,N,3])
    for m in range(M):
        for n in range(N):
            cd = coord.dot(img[m][n])
            new_coord[m][n] += cd
    mean_set = np.zeros([k_mean,2])
    for k in range(k_mean):
        rand_kmean = random_kmean()
        mean_set[k] += rand_kmean    
    # print(f'initail k means: {mean_set}')
    # sys.exit()
    return new_coord,mean_set

def random_kmean():
    L=10
    a = random.randint(0,M-1)
    b = random.randint(0,N-1)
    # print(f"random k means: {[a,b]}")
    # print(f'a,b: {a,b}')
    if a >= L and b >= L:
        k_region = grad_nor[a-L:a+1,b-L:b+1]
        a_l = L
        b_l = L
    elif a < L and b < L:
        k_region = grad_nor[0:a+1,0:b+1]
        a_l = a
        b_l = b
    elif a >= L and b < L:
        k_region = grad_nor[a-L:a+1,0:b+1]
        a_l = L
        b_l = b
    elif a < L and b >= L:
        k_region = grad_nor[0:a+1,b-L:b+1]
        a_l = a
        b_l = L
    P, Q= k_region.shape
    #print(f'P,Q: {P,Q}')
    # for i in range(P):
    #     for j in range(Q):
    #         k_region[i][j] = sum(k_region[i][j]) 
    p= np.unravel_index(k_region.argmin(), k_region.shape)
    k = [p[0]+a-a_l,p[1]+b-b_l] 
    return k    


def check_d(cd,ms):
    global M,N,k_mean,d,ycbcr
    lambda1 = 0.2
    lambda2 = 1.0
    for k in range(k_mean):
        mk = ms[k][0] 
        nk = ms[k][1]
        good_m = list(filter(lambda m: abs(mk-m)<int(M/(k_mean**0.25)),range(M)))
        good_n = list(filter(lambda n: abs(nk-n)<int(N/(k_mean**0.25)),range(N)))
        for m in good_m:
            for n in good_n:
                sec1 = lambda1*((m-mk)**2+(n-nk)**2)
                #sec2 = lambda2*((cd[m][n][0] - ycbcr[k][0])**2 + (cd[m][n][1] - ycbcr[k][1])**2 + (cd[m][n][2] - ycbcr[k][2])**2)
                #sec2 = lambda2*(sum(np.square(cd[m][n] - ycbcr[k])))
                sec2 = lambda2*((cd[m][n][0]-ycbcr[k][0])**2)
                sec3 = (cd[m][n][1]-ycbcr[k][1])**2 +(cd[m][n][2]-ycbcr[k][2])**2
                d[m][n][k] = math.sqrt(sec1+sec2+sec3) 


def find_R():
    global d,R
    for m in range(M):
        for n in range(N):
            h = np.argmin(d[m][n])
            R[m][n] = h
    

def renew_params(nc,ms):
    global k_mean,ycbcr
    new_mean_set = []
    empty_region = []
    B = np.zeros(k_mean)
    #m_pos, n_pos = np.where(R == 1) 
    for k in range(k_mean):
        m_pos, n_pos = np.where(R == k) 

        if len(m_pos) == 0:
            print(f'mean no.{k}\'s region is empty')
            print(f'location of mean {k}: {ms[k]}')
            empty_region.append(k)
            rand_kmean = random_kmean()
            new_mean_set.append(rand_kmean)
        else:
            total = len(m_pos)
            B[k] = total
            m_means = sum(m_pos)/total
            n_means = sum(n_pos)/total
            new_ycbcr = nc[(m_pos,n_pos)].sum(axis = 0)
            # for m in m_pos:
            #     print(f'm : {m}')
            #     for n in n_pos:
            #         new_ycbcr += np.array([nc[m][n][0], nc[m][n][1], nc[m][n][2]])        
            ycbcr[k] = new_ycbcr/total
            new_mean_set.append([m_means,n_means])
    
    return new_mean_set,empty_region,B

def merge_small(nosmall_R):
    global M,N,ycbcr
    # if B(region) < delta merge with adjacent regions

    sm_i, sm_j = np.where(nosmall_R == 0) 
    # print("sm_i,sm_j")
    # print(sm_i,sm_j)
    first_pxl_zero = False
    for v in range(len(sm_i)):
        if sm_i[v] == 0 and sm_j[v] > 0:
            #print("left")
            cmp = nosmall_R[sm_i[v]][sm_j[v]-1]
            if cmp != 0 :
                nosmall_R[sm_i[v]][sm_j[v]] = cmp
            else:
                first_pxl_zero = True            
        elif sm_i[v] > 0 and sm_j[v] == 0:
            #print("up")
            cmp = nosmall_R[sm_i[v]-1][sm_j[v]]
            if cmp != 0:
                nosmall_R[sm_i[v]][sm_j[v]] = cmp
            else:
                first_pxl_zero = True
        else:
            #print("left or up")
            cmp = nosmall_R[sm_i[v]][sm_j[v]-1]
            if cmp != 0 :
                nosmall_R[sm_i[v]][sm_j[v]] = cmp
            else:
                nosmall_R[sm_i[v]][sm_j[v]] = nosmall_R[sm_i[v]-1][sm_j[v]]
    # print(f"first pixel is zero: {first_pxl_zero}")
    return nosmall_R
            
def calc_new_ycbcr(nc,remain_kmean,mergesmall_R):
    global ycbcr
    # print(f"remain_kmean: {remain_kmean[:12]}")
    new_ycbcr = np.zeros([int(remain_kmean.max())+1,3])
    for k in remain_kmean:
        m_pos, n_pos = np.where(mergesmall_R == k) 
        if len(m_pos) == 0:
            print(f'mean no.{k}\'s region is empty')
            print(f'location of mean {k}: {k}')
            raise Exception("region should not be empty")
        else:
            total = len(m_pos)
            # m_means = sum(m_pos)/total
            # n_means = sum(n_pos)/total
            tmp_ycbcr = nc[(m_pos,n_pos)].sum(axis = 0)
            # for m in m_pos:
            #     print(f'm : {m}')
            #     for n in n_pos:
            #         new_ycbcr += np.array([nc[m][n][0], nc[m][n][1], nc[m][n][2]])        
            new_ycbcr[k] = tmp_ycbcr/total
    return new_ycbcr

# merge_R = merge(remain_kmean,mergesmall_R,new_ycbcr)
def merge(remain_kmean,mergesmall_R,new_ycbcr,image):
    global grad_nor,k_mean
    # edge for all remaining mean

    region_pad = np.zeros([M+2,N+2])
    region_pad[1:M+1,1:N+1] = mergesmall_R 
    k_edge_set = []
    k_adj_region = []
    ### consider average grad and laplacian ###
    avg_grad = np.zeros(int(remain_kmean.max()+1)).astype(np.float64)
    L = 6
    sg = 0.3
    pi = np.pi
    n_arr = np.zeros([1,2*L+1])
    lapl = np.zeros([1,2*L+1])
    for i in range(2*L+1):
        # n_arr = [-L ... 0 ... L]
        n_arr[0,i] += i-L
    lapl1 = -pi*sg*np.square(n_arr)
    print(lapl1[0,:11])
    lapl1 = np.exp(lapl1)
    for j in range(2*L+1):
        lapl1[0,j] *= ((2*pi*sg*(j-L))**2 - 2*pi*sg)
    for j in range(2*L+1):
        lapl[0,j] =lapl1[0,j] - (lapl1.sum() / (2*L+1))
    Lpx = convolution.convolution(image,lapl)
    Lpy = convolution.convolution(image,np.transpose(lapl))
    Lp = np.sqrt(np.square(Lpx) + np.square(Lpy))
    print(Lp.shape)
    print("##################################")

    # print(remain_kmean[:10])
    for k in remain_kmean:
        # 2 = up; 3 = left; 4 = down; 5 = right
        edge_set = []
        adj_region = []
        sm_i, sm_j = np.where(region_pad == k) 
        pixel_num = len(sm_i)
        for t in range(pixel_num):
            ### calc average gradient ###
            avg_grad[k] += grad_nor[sm_i[t]-1][sm_j[t]-1]
            
            ### find edge ###
            isEdge = False
            up=left=down=right = 0
            if region_pad[sm_i[t]-1][sm_j[t]]!=k and region_pad[sm_i[t]-1][sm_j[t]]!=0:
                isEdge = True
                up = region_pad[sm_i[t]-1][sm_j[t]]
                if up not in adj_region:
                    adj_region.append(up)
            if region_pad[sm_i[t]][sm_j[t]-1]!=k and region_pad[sm_i[t]][sm_j[t]-1]!=0:
                isEdge = True
                left = region_pad[sm_i[t]][sm_j[t]-1]
                if left not in adj_region:
                    adj_region.append(left)
            if region_pad[sm_i[t]+1][sm_j[t]]!=k and region_pad[sm_i[t]+1][sm_j[t]]!=0:
                isEdge = True
                down = region_pad[sm_i[t]+1][sm_j[t]]
                if down not in adj_region:
                    adj_region.append(down)
            if region_pad[sm_i[t]][sm_j[t]+1]!=k and region_pad[sm_i[t]][sm_j[t]+1]!=0:
                isEdge = True
                right = region_pad[sm_i[t]][sm_j[t]+1]
                if right not in adj_region:
                    adj_region.append(right)
            if isEdge == True:
                grad = grad_nor[sm_i[t]-1][sm_j[t]-1]
                lp = Lp[sm_i[t]-1][sm_j[t]-1]
                edge_set.append([k,grad,up,left,down,right,lp])
        k_edge_set.append(edge_set)
        k_adj_region.append(adj_region)
        avg_grad[k] /= pixel_num
        # index += 1
    largest_mean = int(remain_kmean.max())
    # print(f"score array size: {largest_mean}")
    score = np.zeros([largest_mean+1,largest_mean+1])
    count_mean = 0
    lda1 = 0.5
    lda2 = 0.3
    lda3 = 0.7
    lda4 = 0.3
    lda5 = 0.5
    score_delta = 330
    # shape = new_ycbcr.shape
    # print(f"shape of new ycbcr: {shape}")
    for p in remain_kmean:
        r = 0
        min_score_adj = 100000
        min_adj = 0
        while r < len(k_adj_region[count_mean]):
            q = int(k_adj_region[count_mean][r])
            #print(f"p: {p}")
            #print(f"q: {q}")
            score[p][q] += lda1*(abs(new_ycbcr[p][0] - new_ycbcr[q][0])**2) + lda2*(abs(new_ycbcr[p][1] - new_ycbcr[q][1])**2+abs(new_ycbcr[p][2] - new_ycbcr[q][2])**2)
            k_edges = k_edge_set[count_mean]
            total_grad = 0
            total_lapl = 0
            total = 0
            # e = [k,grad,up,left,down,right]
            for e in k_edges:
                if int(e[0])!=p:
                    raise Exception("Sorry, not right")
                if q in e[2:]:
                    total_grad += e[1]
                    total_lapl += e[6]
                    total += 1
            # print(f"e: {e}")
            # print(f"total grad: {total_grad}")
            score[p][q] += lda3*(total_grad/total)**2
            score[p][q] += lda4*(abs(avg_grad[p]**0.4-avg_grad[q]**0.4))
            score[p][q] += lda5*(total_lapl/total)
            if score[p][q] < min_score_adj:
                min_score_adj = score[p][q]
                min_adj = q
            r += 1
        
        
        if score[p][min_adj] < score_delta:
            # print(score[p][min_adj])
            sm_i, sm_j = np.where(region_pad == p)
            # change to min score adjacent region
            for t in range(len(sm_i)):
                mergesmall_R[sm_i[t]-1][sm_j[t]-1] = min_adj
        count_mean+=1
    return mergesmall_R

def output2(img,merge_R,final_kmean):
    global k_mean
    print('following are output with measure.label')
    #print(f"R: {merge_R[:15,:15]}")
    parent_dir = 'week13'
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
    cv2.imwrite(f'week13/{k_mean}means/output{region}.jpg', img_out)
        


k_means(img2,times)



