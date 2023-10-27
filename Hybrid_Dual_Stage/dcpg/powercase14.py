"""
The main file to run the power system steady state simulation
Inherit from the FDI class
"""

import numpy as np
from pypower.api import case14, ppoption, rundcopf
from pypower.idx_bus import PD, QD
from pypower.idx_brch import RATE_A, BR_X
from config_mea_idx import define_mea_idx_noise
from config_se import se_config, opt
import copy
from DCSE import DCSE
from gen_data import gen_case, gen_load
import scipy.io as scio

class power_env(DCSE):
    def __init__(self, case, case_name, noise_sigma, idx, fpr):
        # Inherit from FDI which inherits from SE 
        super().__init__(case, noise_sigma, idx, fpr)
        
        """
        Modify the grid details
        """
        if case_name == 'case14':
            case['branch'][:,RATE_A] = case['branch'][:,RATE_A]/2.5  # Further reduce the active line flow limit
            small_load_line = [15,18,19]                             # Reduce the active line flow limit in lines with low power loading rate
            case['branch'][small_load_line,RATE_A] = 20
            case['gencost'][1:,4] = 30          # Set the non-ref generator bus linear cost
            
        # Add new attribute
        # self.case = case
        self.f_bus = self.case_int['branch'][:, 0].astype('int')        # list of "from" buses
        self.t_bus = self.case_int['branch'][:, 1].astype('int')        # list of "to" buses
        
        # Test the case
        result = self.run_opf()
        if result['success']:
            print('Initial OPF tests ok.')
        else:
            print('The OPF under this load condition is not converged! Decrease the load.')

        print('*'*60)
        
        # Load
        load_active_dir = f"E:\MY\paper\FDILocation\code\Visual-Power-Grid-master\src\case14\load\load_active.npy"
        load_reactive_dir = f"E:\MY\paper\FDILocation\code\Visual-Power-Grid-master\src\case14\load\load_reactive.npy"

        self.load_active = np.load(load_active_dir)
        self.load_reactive = np.load(load_reactive_dir)        

        # The default reactance in case file
        self.reactance_ori = copy.deepcopy(self.case['branch'][:,BR_X])   
    
    def run_opf(self, **kwargs):
        """
        Rewrite the run_opf function to directly read OPF index
        """
        
        case_opf = copy.deepcopy(self.case)
        if 'opf_idx' in kwargs.keys():
            # print(f'Load index: {10*kwargs["opf_idx"]}.')
            # Specify the index
            load_active_ = self.load_active[int(10*kwargs['opf_idx'])]
            load_reactive_ = self.load_reactive[int(10*kwargs['opf_idx'])]
        
            case_opf['bus'][:,PD] = load_active_
            case_opf['bus'][:,QD] = load_reactive_
        else:
            # run the default
            # print('Run on the default load condition.')
            pass
        
        result = rundcopf(case_opf, opt)
        
        return result
    
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
        
    def se_mtd(self, c, pertub_brh):
        se = copy.deepcopy(self)
        se.update_H(pertub_brh)

        result = se.run_opf(opf_idx=2)
        z, z_noise = se.construct_mea(result)
        a = se.H@c
        a2 = self.H@c
        z_a = np.add(z_noise,a2.reshape(53,1))
        # print(z)

        _, _, r = se.dc_se(z_a)
        # print(z_a)


        return  se, r, a, a2, result
"""
Test
"""
if __name__ == "__main__":
    # Instance power env
    case_name = 'case14'
    case = case14()
    case = gen_case(case, 'case14')  # Modify the case
    
    # Define measurement index
    mea_idx, no_mea, noise_sigma = define_mea_idx_noise(case, 'FULL')
    
    # Instance the class
    case_env = power_env(case = case, case_name = case_name, noise_sigma = noise_sigma, idx = mea_idx, fpr = 0.05)
    
    # Generate load if it does not exist
    _, _ = gen_load(case, 'case14')
    
    # Run opf 
    result = case_env.run_opf()
    
    # Construct the measurement
    z, z_noise = case_env.construct_mea(result) # Get the measurement
    
    # Run DC-SE     
    x_est, z_est, r = case_env.dc_se(z_noise)

    # BDD   
    print(f'BDD threshold: {case_env.bdd_threshold}')
    print(f'residual: {r}')

    print('*'*60)
    print('Run OPF on a given load')
    # Run OPF on a given load
    result = case_env.run_opf(opf_idx = 1)
    gc = result['f']
    # Construct the measurement
    z, z_noise = case_env.construct_mea(result) # Get the measurement   
    x_est, z_est, r = case_env.dc_se(z_noise)
    print(f'BDD threshold: {case_env.bdd_threshold}')
    print(f'residual without attack: {r}')
    # print(f'generator cost without MTD: {gc}')

    # Run single-bus FDI attack
    print('*'*60)
    print('Run single-bus FDI attack')
    za_fdi,a,c = case_env.gen_sin_fdi(z_noise)

    # path = "E:\\MY\\paper\\FDILocation\\code\\data\\case14\\single"
    # c_ori_mat = scio.loadmat(path+"\c_0.mat")['c']
    # c_ori = c_ori_mat[:,:,0]
    # c = c_ori[0]
    # a = case_env.H@c
    # za_fdi = np.add(z_noise,a.reshape(53,1))
    x_fdi, z_fdi, r_fdi = case_env.dc_se(za_fdi)

    print(f'residual with FDI attack: {r_fdi}')

    print('*'*60)
    print('Run MTD')
    # Run MTD
    pos = np.argmax(abs(c))
    attack_bus = case_env.non_ref_index[pos]
    brh = 0
    for i in range(case_env.no_brh): # Find the branch connected to the attacked bus
        f = case_env.f_bus[i]
        t = case_env.t_bus[i]
        if f==attack_bus or t==attack_bus:
            brh = i
            break
    se, r_mtd ,_,_,result_mtd  = case_env.se_mtd(c,brh)
    # gc_mtd = result_mtd['f']
    # se, r_mtd = case_env.se_mtd(c,4)
    print(f'residual with FDI attack and MTD: {r_mtd}')
    # print(f'generator cost after MTD: {gc_mtd}')
    # path = "E:\MY\paper\FDILocation\code\data\case14"
    # z_sum5 = scio.loadmat(path+"\\z_0.mat")['z'][:,:,0]
    # z =np.expand_dims(z_sum5[1], axis = 1)
    # x_est, z_est, r = case_env.dc_se(z)
    # print(r)
