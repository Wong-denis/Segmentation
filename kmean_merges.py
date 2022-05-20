import numpy as np
import convolution
from math import sqrt,log10


def merge_small(nosmall_R, size, ycbcr):
    # global M,N,ycbcr
    # if B(region) < delta merge with adjacent regions
    M,N = size[0],size[1]
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
            
def calc_new_ycbcr(nc,remain_kmean,mergesmall_R, ycbcr):
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
def merge(remain_kmean,mergesmall_R,new_ycbcr,image, grad_nor, k_mean, min_region_area):
    # global grad_nor,k_mean
    global region_pad, k_edge_set, k_adj_region
    size = image.shape
    M,N = size[0],size[1]

    # edge for all remaining mean
    region_pad = np.zeros([M+2,N+2])
    region_pad[1:M+1,1:N+1] = mergesmall_R 
    k_edge_set = []
    k_adj_region = []

    ### calculate laplacian ###
    calc_laplacian(remain_kmean, image, L=6, sg=0.3)
    ### calculate adjacent region and avg_grad
    k_edge_set, k_adj_region = adjacent_regions(remain_kmean, grad_nor)

    largest_mean = int(remain_kmean.max())
    print(f"largest mean: {largest_mean}")
    # print(f"score array size: {largest_mean}")
    score = np.zeros([largest_mean+1,largest_mean+1])
    count_mean = 0
    lda0 = 0.5
    lda1 = 0.3
    lda2 = 0.7
    lda3 = 0.3
    lda4 = 0.5
    lda = [lda0,lda1,lda2,lda3,lda4]
    score_delta = np.zeros(largest_mean+1)
    # score_delta = adaptive_delta(score_delta)
    # shape = new_ycbcr.shape
    # print(f"shape of new ycbcr: {shape}")
    denom = kmean_denom_curve(k_mean)
    delta_lda0 = calc_delta_lda0(denom)
    print(f"denominator: {denom} ; lda0: {delta_lda0}")
    for p in remain_kmean:
        min_adj,score = calc_score(p,lda, score, new_ycbcr,count_mean)
        score_delta[p] = adaptive_delta(p,min_adj,min_region_area,denom,delta_lda0,count_mean)
        if score[p][min_adj] < score_delta[p]:
            # print(score[p][min_adj])
            sm_i, sm_j = np.where(region_pad == p)
            # change to min score adjacent region
            for t in range(len(sm_i)):
                mergesmall_R[sm_i[t]-1][sm_j[t]-1] = min_adj
        count_mean+=1
    return mergesmall_R

def adaptive_delta(p, min_adj, min_area,  denom, delta_lda0,count_mean, start_delta =400.0,min_delta =200.0, max_delta =500.0):
    # denominator
    delta_lda1 = 5
    delta = start_delta
    a = min(regions_area[p],regions_area[min_adj])
    reduce = log10((a/denom) + (1-(min_area/denom))) * delta_lda0
    #print(f"reduce: {reduce}")
    delta -= reduce
    if delta < 200:
        print(f"first delta: {delta}")
        #print(f"area: {a}")
    Length = len(k_edge_set[count_mean])
    b = min(sqrt(regions_area[p]),sqrt(regions_area[min_adj]))
    # print(f"Length/sqrt(area): {Length/b}")
    delta += (Length/b) * delta_lda1
    # print(f"second delta: {delta}")
    # avg_grad usually < 80, >10
    # print(f"min average grad: {min(avg_grad[p],avg_grad[min_adj])}")
    c = min(avg_grad[p],avg_grad[min_adj])
    delta += 4*(1.5**(c/10))
    print(f"second delta: {delta}")
    if delta > max_delta:
        delta = max_delta
    elif delta < min_delta:
        delta = min_delta
    return delta

def kmean_denom_curve(k_mean):
    return 30*(0.95**((k_mean/2)-25))

def calc_delta_lda0(denom):
    return 230/(log10(2000/denom))

def calc_score(p,lda,score,new_ycbcr,count_mean):
    r = 0
    min_score_adj = 100000
    min_adj = 0
    while r < len(k_adj_region[count_mean]):
        q = int(k_adj_region[count_mean][r])
        #print(f"p: {p}")
        #print(f"q: {q}")
        score[p][q] += lda[0]*(abs(new_ycbcr[p][0] - new_ycbcr[q][0])**2) + lda[1]*(abs(new_ycbcr[p][1] - new_ycbcr[q][1])**2+abs(new_ycbcr[p][2] - new_ycbcr[q][2])**2)
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
        score[p][q] += lda[2]*(total_grad/total)**2
        score[p][q] += lda[3]*(abs(avg_grad[p]**0.4-avg_grad[q]**0.4))
        score[p][q] += lda[4]*(total_lapl/total)
        if score[p][q] < min_score_adj:
            min_score_adj = score[p][q]
            min_adj = q
        r += 1
    return min_adj,score

def calc_laplacian(remain_kmean, image, L =6, sg =0.3):
    global avg_grad,Lp
    avg_grad = np.zeros(int(remain_kmean.max()+1)).astype(np.float64)
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

def adjacent_regions(remain_kmean, grad_nor):
    global regions_area
    regions_area = np.zeros(int(remain_kmean.max())+1)
    for k in remain_kmean:
        # 2 = up; 3 = left; 4 = down; 5 = right
        edge_set = []
        adj_region = []
        sm_i, sm_j = np.where(region_pad == k) 
        regions_area[k] = len(sm_i)
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
    print(f"Max Area: {regions_area.max()}")
    return k_edge_set, k_adj_region