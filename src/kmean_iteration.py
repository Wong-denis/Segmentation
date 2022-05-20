import numpy as np
import random
import math

def connected(remain_kmean,merge_R,size):
    M,N = size[0],size[1]
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

def initial(img, size, k_mean, grad_nor, L =10):
    #global grad_nor
    M,N = size[0],size[1]
    coord = np.array([[0.114,0.587,0.299],[0.5,-0.331,-0.169],[-0.081,-0.419,0.5]])
    new_coord = np.zeros([M,N,3])
    for m in range(M):
        for n in range(N):
            cd = coord.dot(img[m][n])
            new_coord[m][n] += cd
    mean_set = np.zeros([k_mean,2])
    for k in range(k_mean):
        rand_kmean = random_kmean(grad_nor, L, size)
        mean_set[k] += rand_kmean    
    # print(f'initail k means: {mean_set}')
    # sys.exit()
    return new_coord,mean_set

def random_kmean(grad_nor, L, size):
    M,N = size[0], size[1]
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
    # P, Q= k_region.shape
    #print(f'P,Q: {P,Q}')
    # for i in range(P):
    #     for j in range(Q):
    #         k_region[i][j] = sum(k_region[i][j]) 
    p= np.unravel_index(k_region.argmin(), k_region.shape)
    k = [p[0]+a-a_l,p[1]+b-b_l] 
    return k    

def check_d(cd, ms, size, k_mean, d, ycbcr):
    # global M,N,k_mean,d,ycbcr
    M,N = size[0], size[1]
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

def find_R(size, d, R):
    # global d,R
    M, N = size[0], size[1]
    for m in range(M):
        for n in range(N):
            h = np.argmin(d[m][n])
            R[m][n] = h
    

def renew_params(nc,ms, k_mean, ycbcr, R, random_params,L=10):
    #global k_mean,ycbcr
    grad_nor, size = random_params[0],random_params[1]
    new_mean_set = []
    # empty_region = []
    #m_pos, n_pos = np.where(R == 1) 
    for k in range(k_mean):
        m_pos, n_pos = np.where(R == k) 

        if len(m_pos) == 0:
            print(f'mean no.{k}\'s region is empty')
            print(f'location of mean {k}: {ms[k]}')
            # empty_region.append(k)
            rand_kmean = random_kmean(grad_nor, L, size)
            new_mean_set.append(rand_kmean)
        else:
            total = len(m_pos)
            m_means = sum(m_pos)/total
            n_means = sum(n_pos)/total
            new_ycbcr = nc[(m_pos,n_pos)].sum(axis = 0)
            # for m in m_pos:
            #     print(f'm : {m}')
            #     for n in n_pos:
            #         new_ycbcr += np.array([nc[m][n][0], nc[m][n][1], nc[m][n][2]])        
            ycbcr[k] = new_ycbcr/total
            new_mean_set.append([m_means,n_means])
    
    return new_mean_set
