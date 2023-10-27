import pickle
import numpy as np
import scipy.io as sio
from DCSE import DCSE
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *

from sklearn import metrics
from itertools import combinations
from collections import defaultdict
import copy

def connected(self,a,b):
    f = np.where(self.f_bus == a)[0]
    t = np.where(self.t_bus == a)[0]

    if len(f) != 0:
        for i in f:
            if b == self.t_bus[i]:
                return 1
    if len(t) != 0:
        for i in t:
            if b == self.f_bus[i]:
                return 1
    return 0

def is_leaf(self,node):
    node_list = list(self.f_bus) + list(self.t_bus)
    flag = node_list.count(node)
    if flag == 1:
        return 1
    else:
        return 0
    
def c_bool(c):
    """
    The re-estimated injection phase angle c from decomposition：
    Convert the bus that are attacked to 1, the bus that are not attacked to 0
    """

    c = [abs(i) for i in c]
    # c_new = [abs(i) for i in c]
    # c_new.pop(c.index(max(c)))
    # c_new.pop(c.index(max(c)))
    c_attacked = []

    for i in c:
        if abs(i) >= 0.1:
            c_attacked.append(1)
        else:
            c_attacked.append(0)
    return c_attacked

def c_est_convert(self,c):
    """
    The re-estimated injection phase angle c from decomposition：
    Convert the bus that are attacked to 1(coordinate is 2, coordinate flag is 3), the bus that are not attacked to 0
    """

    c = [abs(i) for i in c]
    c_new = [abs(i) for i in c]
    # c_new.pop(c.index(max(c)))
    # c_new.pop(c.index(max(c)))
    c_attacked = []
    c_dict = defaultdict(list)
    # c_std = np.std(c_new)
    # c_mean = np.mean(c_new)

    for i in c:
        if abs(i) >= 0.02:
            c_attacked.append(round(i, 3))
        else:
            c_attacked.append(0)
    for i,j in enumerate(c_attacked):
        c_dict[j].append(i)
    for i in c_dict.keys():                    
        if i!=0:
            if len(c_dict[i])>1:
                for j in combinations(c_dict[i],2):
                    if connected(self,j[0],j[1]):
                        if is_leaf(self,j[0])==0 and c_attacked[j[1]]!=2:
                            c_attacked[j[0]] = 3
                            c_attacked[j[1]] = 2
                        elif is_leaf(self,j[1])==0 and c_attacked[j[0]]!=2:
                            c_attacked[j[1]] = 3
                            c_attacked[j[0]] = 2
                    else:
                        if c_attacked[j[0]] != 2 and c_attacked[j[0]] != 0:
                            c_attacked[j[0]] =1
                        if c_attacked[j[1]] != 2 and c_attacked[j[1]] != 0:
                            c_attacked[j[1]] =1
            elif len(c_dict[i])==1:
                c_attacked[c_dict[i][0]] =1
    return c_attacked

def gen_fdi(self, z, c):
    """
    Single bus / random value / FDI attack
    """
    # c_this = copy.deepcopy(c)
    # c_this.pop(0)
    # Attack vector
    a = self.H@c
    z_a = z + a

    return z_a, a

def update_H(self, brh):
    """
    Update H of self
    """
    se = copy.deepcopy(self)
    increase_decrease = np.random.randint(0, 2)*2-1   # -1 or 1
    # ratio = np.ones(self.no_brh,)
    ratio = (se.min_reac_ratio + (se.max_reac_ratio - se.min_reac_ratio) * np.random.rand())*increase_decrease
    # ratio = 8
    se.x[brh] = se.x[brh] * (1+ratio)

    se.case['branch'][:,BR_X] = se.x
    se.case_int = ext2int(self.case)
    
    # reactance
    se.neg_b = -1/se.x

    # susceptance matrix
    se.D = np.diag(se.neg_b)
    se.B = se.Ar.T@np.diag(se.neg_b)@se.Ar
    se.S = np.diag(se.neg_b)@se.Ar

    # measurement matrix
    se.H = np.vstack((se.B,np.vstack((se.S,-se.S))))

    return se

def perturb_strategy(self, c):
    att_bus = []
    for i in range(len(c)):
        if c[i] != 0:
            att_bus.append(i)

    brh = [i for i in range(self.no_brh)]
    pertub_brh = []
    flag = 0

    for i in range(self.no_brh):
        if self.f_bus[i] in att_bus and self.t_bus[i] in att_bus:
            brh.pop(flag)
        else:
            flag = flag + 1
    # print(brh)
    for i in brh:
        if self.f_bus[i] in att_bus:
            if c[self.f_bus[i]] !=2:
                pertub_brh.append(i)
            att_bus.remove(self.f_bus[i])
        elif self.t_bus[i] in att_bus:
            if c[self.t_bus[i]] !=2:
                pertub_brh.append(i)
            att_bus.remove(self.t_bus[i])
    return pertub_brh

def se_mtd(self, c, c_attacked, pertub_brh):
    for i in pertub_brh:
        se = copy.deepcopy(self)
        se = update_H(se, i)
        opt = ppoption()              # OPF options
        opt['VERBOSE'] = 0
        opt['OUT_ALL'] = 0
        opt['OPF_FLOW_LIM'] = 1

        if se.f_bus[i] != 117 and se.t_bus[i] != 117:
            result = rundcopf(se.case, opt)
            z, z_noise = DCSE.construct_mea(se, result)
            za, a = gen_fdi(self, z, c)
            za2, a2 = gen_fdi(se, z, c)


            if list(a) == list(a2):
                if c_attacked[se.f_bus[i]] == 3 or c_attacked[se.t_bus[i]] == 3:
                    for i in c_attacked:
                        if i == 2 or i == 3:
                            i = 0
                elif c_attacked[se.f_bus[i]] == 1:
                    c_attacked[se.f_bus[i]] = 0
                elif c_attacked[se.t_bus[i]] == 1:
                    c_attacked[se.t_bus[i]] = 0

    return c_attacked

def mtd(se, c_true, c_attacked, times):
    c_attacked_cvt = np.zeros((times,117))
    # c_true_cvt = np.zeros((times,117))
    c_sure = np.zeros((times,117))
    for i in range(times):
        c_attacked_cvt[i] = c_est_convert(se, c_attacked[i])
        # c_true_cvt[i] = c_est_convert(se, c_true[i])
        pertub_brh = perturb_strategy(se, c_attacked_cvt[i])
        c_sure[i] = se_mtd(se, c_true[i], c_attacked_cvt[i], pertub_brh)
        c_sure[i] = c_bool(c_sure[i])
    return c_sure
    


def fpr_tpr(y_true, y_pred):
    y_true = y_true.reshape(11700, 1)
    y_pred = y_pred.reshape(11700, 1)

    tn, fp, fn, tp, = metrics.confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)

    return fpr,tpr


if __name__ == "__main__":

    with open("E:\\MY\\paper\\FDILocation\\code\\data\\se_test.pkl", "rb") as f:
        se = pickle.loads(f.read())

    path = "E:\\MY\\paper\\FDILocation\\code\\data\\comul"
    c_omogmf_mat = sio.loadmat(path+"\C_new\omogmf\c_new-5%.mat")['c_new']
    c_godec_mat = sio.loadmat (path+"\C_new\godec\c_new-5%.mat")['c_new']
    c_LRMF_mat = sio.loadmat (path+"\C_new\LRMF\c_new-5%.mat")['c_new']
    # original data
    c_ori_mat = sio.loadmat(path+"\C\c-5%.mat")['c']
    c_ori = c_ori_mat[:,:,0]

    c_omogmf_cvt = np.zeros((100,117))
    c_godec_cvt = np.zeros((100,117))
    c_LRMF_cvt = np.zeros((100,117))
    c_ori_cvt = np.zeros((100,117))
    
    for i in range(100):
        c_omogmf_cvt[i] = c_bool(c_omogmf_mat[i])
        c_godec_cvt[i] = c_bool(c_godec_mat[i])
        c_LRMF_cvt[i] = c_bool(c_LRMF_mat[i])
        c_ori_cvt[i] = c_bool(c_ori[i])

    c_omogmf_sure = mtd(se, c_ori, c_omogmf_mat, 100)
    c_LRMF_sure = mtd(se, c_ori, c_LRMF_mat, 100)
    c_godec_sure = mtd(se, c_ori, c_godec_mat, 100)

    omogmf_fp, omogmf_tp = fpr_tpr(c_ori_cvt,c_omogmf_sure)
    godec_fp, godec_tp = fpr_tpr(c_ori_cvt,c_godec_sure)
    LRMF_fp, LRMF_tp = fpr_tpr(c_ori_cvt,c_LRMF_sure)

    omogmf_fp_ori, x = fpr_tpr(c_ori_cvt,c_omogmf_cvt)
    godec_fp_ori, y = fpr_tpr(c_ori_cvt,c_godec_cvt)
    LRMF_fp_ori, z = fpr_tpr(c_ori_cvt,c_LRMF_cvt)




    print("hi")


