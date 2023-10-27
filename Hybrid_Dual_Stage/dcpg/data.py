# import numpy as np
from pypower.api import case14
from pypower.idx_bus import PD, QD
from pypower.idx_brch import RATE_A, BR_X
from config_mea_idx import define_mea_idx_noise
# from config_se import se_config, opt
# import copy
from DCSE import DCSE
from gen_data import gen_case, gen_load
from powercase14 import power_env
import scipy.io as scio

def dcopf_normal(case_env, times, noise_flag):
    z_sum = []
    # r_sum = []
    print(f'Generating normal data, noise value: {noise_flag}\%')

    for i in range(times):
        result = case_env.run_opf(opf_idx = i)
        # print(f'Is {i}th OPF success: {result["success"]}')
        z, z_noise = case_env.construct_mea(result) # Get the measurement 
        if noise_flag > 0:
            z_mea = z_noise
        else:
            z_mea = z
        z_sum.append(z_mea)
    return z_sum


def dcopf_attack(case_env, times, noise_flag, att_type):
    za_sum = []
    a_sum = []
    c_sum = []
    print(f'Generating attack data, attack type: {att_type}, noise value: {noise_flag}\%')

    for i in range(times):
        result = case_env.run_opf(opf_idx = i+5)
        # print(f'Is OPF success: {result["success"]}')
        z, z_noise = case_env.construct_mea(result) # Get the measurement
        if noise_flag > 0:
            z_mea = z_noise
        else:
            z_mea = z
        
        if att_type == 'single':#single-bus
            za_fdi,a,c = case_env.gen_sin_fdi(z_mea)
        elif att_type == 'unmul': #uncoordiante multiple-bus
            za_fdi,a,c = case_env.gen_mul_fdi(z_mea)
        elif att_type == 'comul': #coordiante multiple-bus
            za_fdi,a,c = case_env.gen_co_fdi(z_mea)

        za_sum.append(za_fdi)
        a_sum.append(a)
        c_sum.append(c)
    return za_sum, a_sum, c_sum



if __name__ == "__main__":
    # Instance power env
    case_name = 'case14'
    case = case14()
    case = gen_case(case, 'case14')  # Modify the case
    
    # Define measurement index
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    
    # Generate load if it does not exist
    _, _ = gen_load(case, 'case14')
    
    # Instance the class
    case_env = power_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
    
    # data_config
    noise_flag = 5
    att_flag = 1
    att_type = 'comul'
    att_times = 100
    no_att_bus = 5
    

    # saving_path
    z_dir = f"E:\MY\paper\FDILocation\code\data\case14\z_{noise_flag}-pbt.mat"
    za_dir = f"E:\MY\paper\FDILocation\code\data\case14\{att_type}\za_{noise_flag}-pbt{no_att_bus}.mat"
    a_dir = f"E:\MY\paper\FDILocation\code\data\case14\{att_type}\\a_{noise_flag}-pbt{no_att_bus}.mat"
    c_dir = f"E:\MY\paper\FDILocation\code\data\case14\{att_type}\c_{noise_flag}-pbt{no_att_bus}.mat"

    if att_flag:
        za_sum, a_sum, c_sum = dcopf_attack(case_env, att_times, noise_flag, att_type)
        scio.savemat(za_dir, {'za': za_sum})
        scio.savemat(a_dir, {'a': a_sum})
        scio.savemat(c_dir, {'c': c_sum})
    else:
        z_sum = dcopf_normal(case_env, att_times, noise_flag)
        scio.savemat(z_dir, {'z': z_sum})

    # scio.savemat("E:\MY\paper\FDILocation\code\data\case14\H.mat", {'h': case_env.H})
    # 
    


