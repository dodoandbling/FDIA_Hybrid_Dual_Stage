import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_brch import *
from pypower.idx_gen import *

from sklearn import metrics
from itertools import combinations
from collections import defaultdict
import copy
import scipy.io as scio

from powercase14 import power_env
from config_mea_idx import define_mea_idx_noise
from gen_data import gen_case, gen_load

def get_iqr_data(datas):
    q1=np.quantile(datas,0.25)
    q2=np.median(datas)
    q3=np.quantile(datas,0.75)
    iqr=q3-q1
    up=q3+1.5*iqr
    return up

def c_bool(c):
    """
    The re-estimated injection phase angle c from decomposition：
    Convert the bus that are attacked to 1, the bus that are not attacked to 0
    """
    c_cvt = np.zeros((c.shape[0],c.shape[1]))

    for i in range(c.shape[0]):
        # c[i] = [abs(j) for j in c[i]]
        c_attacked = []
        c_abs = list(map(abs,c[i]))
        up = get_iqr_data(c_abs)
        for j in c_abs:
            if j > up:
                c_attacked.append(1)
            else:
                c_attacked.append(0)
        c_cvt[i] = c_attacked
    return c_cvt

def fpr_tpr(y_true, y_pred):
    [m,n] = y_pred.shape
    y_true = y_true.reshape(m*n, 1)
    y_pred = y_pred.reshape(m*n, 1)

    m = metrics.confusion_matrix(y_true, y_pred).ravel()

    if len(m)>4:
        cl = m[4]
        fr = m[3]
        ab = np.count_nonzero(y_true==1)
        nb = np.count_nonzero(y_true==0)

        tpr = cl / ab
        fpr = fr / nb
    else:
        fpr = m[1] / (m[1] + m[0])
        tpr = m[3] / (m[3] + m[2])

    return fpr,tpr

class mtd():
    def __init__(self, case_env):
        self.no_mea = case_env.no_mea
        # self.ca_sure = np.zeros((times,no_mea))
        self.env = case_env


    def att_verify(self, c):
        no_brh = self.env.no_brh
        no_bus = len(c)
        c_located = [2] * no_bus
        # brh = [i for i in range(self.env.no_brh)]
        brh = [4,5,6,7,8,9,10,11,13,15,16,18]
        # flag = 0
        # for i in range(no_brh):
        #     if self.env.f_bus[i] == self.env.ref_index or self.env.t_bus[i] == self.env.ref_index:
        #         brh.pop(flag)
        #     else:
        #         flag = flag + 1
        bus_alert = [0] * len(brh)
        for i in brh:
            _, _, a, a2, _ = self.env.se_mtd(c, i)
            if list(a) != list(a2):
                bus_alert[brh.index(i)] = 1

        for i in range(len(bus_alert)):
            if bus_alert[i] == 0:
                index = brh[i]
                f_bus = self.env.f_bus[index]
                t_bus = self.env.t_bus[index]
                f_bus_index = self.env.non_ref_index.index(f_bus)
                t_bus_index = self.env.non_ref_index.index(t_bus)
                c_located[f_bus_index] = 0
                c_located[t_bus_index] = 0
        for i in range(len(bus_alert)):
            if bus_alert[i] == 1:
                index = brh[i]
                f_bus = self.env.f_bus[index]
                t_bus = self.env.t_bus[index]
                f_bus_index = self.env.non_ref_index.index(f_bus)
                t_bus_index = self.env.non_ref_index.index(t_bus)
                if c_located[f_bus_index] == 0:
                    c_located[t_bus_index] = 1
                elif c_located[t_bus_index] == 0:
                    c_located[f_bus_index] = 1
        return c_located
    
    
    def att_verify_loop(self, c_true, times):
        c_sure = np.zeros((times,self.env.no_non_ref))
        # r_new_sum = []
        # pbrh_time_sum = []
        for i in range(times):
            c_sure_i = self.att_verify(c_true[i])
            c_sure[i,:] = c_sure_i
            # r_new_sum.append(r)
        # return c_sure, r_new_sum
        return c_sure
    
    
    def pbtime_loop(self, c_attacked, times):
        pbrh_time_sum = []
        for i in range(times):
            c_attacked_cvt = self.c_est_convert(c_attacked[i])
            pertub_brh = self.perturb_strategy(c_attacked_cvt)
            pbrh_time_sum.append(len(pertub_brh))
        return pbrh_time_sum



if __name__ == "__main__":
    # Instance power env
    case_name = 'case14'
    case = case14()
    case = gen_case(case, 'case14') 
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    case_env = power_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
    _, _ = gen_load(case, 'case14')

    path = "E:\\MY\\paper\\FDILocation\\code\\data\\case14"
    c_ori_mat = scio.loadmat(path+"/single/c_10.mat")['c']
    c_sin0_true_sum = c_ori_mat[:,:,0]
    # c_new_sin0_sum = c_new_sin0_mat[:,:,0]

    c_sin0_true_bool = c_bool(c_sin0_true_sum)
         
    pb = mtd(case_env=case_env)
    c_sure_sin0_sum = pb.att_verify_loop(c_sin0_true_sum, 50)
    # print(c_sure)
    # c_sure_sin0_bool = c_bool(c_sure_sin0_sum)

    sin0_fp_mtd, sin0_tp_mtd = fpr_tpr(c_sin0_true_bool[0:50 ,:], c_sure_sin0_sum)

    print(f'tp/fp of PBSonly:{sin0_tp_mtd}/{sin0_fp_mtd}')



#     print("hi")


