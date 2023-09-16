import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from makeGrid import generateGrid, genDict


class ReinerModel:
    def __init__(self, data, N_X, N_Y, alpha, beta, gamma, T):
        self.data=data
        self.N_X=N_X
        self.N_Y=N_Y
        self.alpha=alpha
        self.beta = beta
        self.gamma=gamma
        self.T = T
        self.break_while_flag=0
        self.center_index=(N_Y+1)*N_X/2 
        self.boundary=int(N_Y*0.4)-1

        for i in self.data.keys():
            self.data[i]['state']=beta
            self.data[i]['add_con_state'] = 0
            self.data[i]['nonreceptive_state'] = 0
            self.data[i]['avg_state'] = 0
        self.data[self.center_index]['state']=1
        
        
    def update_one_step(self):
        for i in range(N_X * N_Y):
            # update the add constant state
            flag = 0
            if self.data[i]['state'] >= 1:
                flag = 1
            else:
                for j in self.data[i]['ring1']:
                    if self.data[j]['state'] >= 1:
                        flag = 1
                        break
            self.data[i]['add_con_state'] = flag * (self.data[i]['state']+self.gamma)
            
            # update the nonreceptive state
            if flag == 1:
                self.data[i]['nonreceptive_state'] = 0
            else:
                self.data[i]['nonreceptive_state'] = self.data[i]['state']
        
        for i in range(N_X * N_Y):
            # update the average state
            neighbor_add_up = sum(self.data[j]['nonreceptive_state'] for j in self.data[i]['ring1'])
            self.data[i]['avg_state'] = (1-0.5*self.alpha)*self.data[i]['nonreceptive_state'] + \
                self.alpha/(2*len(self.data[i]['ring1']))*neighbor_add_up
            
            # update the state
            self.data[i]['state'] = self.data[i]['add_con_state'] + self.data[i]['avg_state']
            
            
    def automatic_stop(self):
        flag = False
        boundary_li = []
        boundary_li.extend(range(self.N_Y))
        boundary_li.extend(range((self.N_X-1)*self.N_Y, self.N_X*self.N_Y))   
        boundary_li.extend(range(0, self.N_X*self.N_Y, self.N_Y))         
        boundary_li.extend(range(self.N_X-1, self.N_X*self.N_Y, self.N_Y))   
        flag = True
        for i in boundary_li:
            if abs(self.data[i]['state']-self.beta)>0.01*self.beta:
                flag = False
                print(i)
                break
        return flag
        
    
    def update_n_step(self, n_iter=100):
        # for i in range(n_iter):
        #     self.update_one_step()
        
        while self.automatic_stop():
            self.update_one_step()
        
        # Some values are greater than 1, turn them to 1
        for i in range(N_X * N_Y):
            if self.data[i]['state'] > 1:
                self.data[i]['state'] = 1
                
        # make the plot
        plt.style.use('dark_background')
        fig_width = 20
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))

        for i in range(self.N_X*self.N_Y):
            hex = RegularPolygon((self.data[i]['pos'][0], self.data[i]['pos'][1]), numVertices=6, radius=1/np.sqrt(3), 
                                 orientation=np.radians(0),  facecolor = (1,1,1),
                                 alpha=min(1,self.data[i]['state']))#, edgecolor='k')
            ax.add_patch(hex)
        ax.axis('equal')
        # plt.savefig('../image/reiter_a_{}_b_{}_g_{}.png'.format(self.alpha, self.beta, self.gamma))
        # plt.savefig('../image/reiter_T_{}.png'.format(self.T))
        # plt.close()

def compute_P(T):
    return 1-0.001*T/0.65


def compute_alpha(T, P):
    # print(((T+273.15)**1.5/(P+ 0.000001)- 4514.418)/890.190)
    return -((T+273.15)**1.5/(P+ 0.000001) - 4514.418)/890.190#2531.681

def compute_gamma(T,p_h2o):
    # T (degree C)
    # p_h2o (hPa = 1000 * bar value)
    p_sat = 6.1115*np.exp((23.036 - T/333.7)*(T/(279.82 + T)))
    return p_h2o/p_sat/10

def compute_beta(T, gamma):
    return gamma * (-T)/(30) + 0.5


if __name__=="__main__":

    N_X = 100
    N_Y = N_X
    X, Y = generateGrid(N_X, N_Y)
    data = genDict(X, Y)
    
    # beta_li = [0.01,0.3, 0.5, 0.7, 0.9, 1]
    # gamma_li = [0.0001, 0.001, 0.01, 0.1, 1, 10]

    # beta_li = [0.5, 0.7]
    # gamma_li = [0.0, 0.0035]
    for T in range(-15,1):
        P = compute_P(T)
        # print(T, P)
        alpha = compute_alpha(T,P)
        gamma = compute_gamma(T, p_h2o=0.01)
        beta = compute_beta(T, gamma)
        print(alpha, gamma, beta)
    # for beta in beta_li:
        # for gamma in gamma_li:
        model=ReinerModel(data, N_X, N_Y, alpha=alpha, beta=beta, gamma=gamma, T = T)
        model.update_n_step()
        # break

    
