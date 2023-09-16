import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as sk
from matplotlib.patches import RegularPolygon


# make grid
def generateGrid(N_X, N_Y): 
    # input: number of grids on x (horizontal) and y (vertical) direction
    # generate grids with spacings = 1
    X, Y = np.meshgrid(np.arange(N_X), np.arange(N_Y), sparse=False)
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Y = Y*np.sqrt(3)/2
    X[::2, :] += 1/2
    
    return X, Y




def genDict(X, Y):
    # return dictionary of each grid point containing positions, first- and second-nearest neighbor lists and crystallization states
    
    pos = [[x,y] for x, y in zip(X.flatten(), Y.flatten())] # list of grids
    tree = sk.KDTree(pos)
    idx_dist1 = tree.query_radius(pos, r=1.5) 
    idx_dist2 = tree.query_radius(pos, r=2.5) 


    for k in range(len(pos)):
        idx_dist1[k] = list(set(idx_dist1[k]) - set([k]))  # first ring
        idx_dist2[k] = list(set(idx_dist2[k]) - set(idx_dist1[k]) - set([k])) # second ring
    
    
    states = [0]*len(pos)
    d = {'pos': pos, 'ring1': idx_dist1, 'ring2': idx_dist2, 'state': states}
    df = pd.DataFrame(data=d)
    data = df.to_dict('index')    
    
    return data


def Visualize(i, j, data, alpha, group=None):
    
    # visualize specific grid(s)
    # i: x index (x from 0 to N_X-1)
    # j: y index (y from 0 to N_Y-1)
    # group: None or ring1 or ring2
    
    idx = N_Y*j + i
    
    if group is not None:
        for neighbor in data[idx][group]:
            hex = RegularPolygon((data[neighbor]['pos'][0], data[neighbor]['pos'][1]), numVertices=6, radius=1/np.sqrt(3), 
                                 orientation=np.radians(0),  facecolor = (1,1,1),
                                 alpha=alpha, edgecolor='k')
            ax.add_patch(hex)
    else:
        hex = RegularPolygon((data[idx]['pos'][0], data[idx]['pos'][1]), numVertices=6, radius=1/np.sqrt(3), 
                             orientation=np.radians(0),  facecolor = (1,1,1),
                             alpha=alpha, edgecolor='k')
        ax.add_patch(hex)
        
    return

class GridPoint:
    def __init__(self, data, i, j):
        idx = N_Y*j + i
        self.pos = data[idx]['pos']
        self.ring1 = data[idx]['ring1']
        self.ring2 = data[idx]['ring2']
        self.state = data[idx]['state']
        
        
# #neighbors
# if i%2==0:
#     ax.scatter(X[i+1,j], Y[i+1,j], s=50, c='g')
#     ax.scatter(X[i+1,j+1], Y[i+1,j+1], s=50, c='g')
#     ax.scatter(X[i,j+1], Y[i,j+1], s=50, c='g')
#     ax.scatter(X[i,j-1], Y[i,j-1], s=50, c='g')
#     ax.scatter(X[i-1,j], Y[i-1,j], s=50, c='g')
#     ax.scatter(X[i-1,j+1], Y[i-1,j+1], s=50, c='g')
    
# else:
#     ax.scatter(X[i+1,j], Y[i+1,j], s=50, c='g')
#     ax.scatter(X[i+1,j-1], Y[i+1,j-1], s=50, c='g')
#     ax.scatter(X[i,j+1], Y[i,j+1], s=50, c='g')
#     ax.scatter(X[i,j-1], Y[i,j-1], s=50, c='g')
#     ax.scatter(X[i-1,j], Y[i-1,j], s=50, c='g')
#     ax.scatter(X[i-1,j-1], Y[i-1,j-1], s=50, c='g')

###### 3-equation update method #######

class three_equation_snowflake_generation:
    def __init__(self, data, N_X, N_Y, beta, gamma, timestep, T):
        self.data=data
        self.T = T
        self.N_X=N_X
        self.N_Y=N_Y
        self.beta=beta
        self.gamma=gamma
        self.break_while_flag=0
        self.timestep=timestep
        self.center_index=int(N_Y/2)*N_X+int(N_X/2)
        self.boundary=int(N_Y*0.4)
        
        self.current_state=0
        self.updated_state=0
        self.crystalized_flag_ring1=0
        self.crystalized_flag_ring2=0
        for i in self.data.keys():
            self.data[i]['state']=self.beta
        self.data[self.center_index]['state']=1
        
    def if_crystalized_in_circle(self,circle_pos_list):
        crystalized_flag=0
        for i in circle_pos_list:
            if self.data[i]['state']>=1:
                crystalized_flag=1
                break
        return crystalized_flag
        
    def decide_receptive(self):
        #1: non-receptive < 1
        #0: receptive >=1
        
        for i in self.data.keys():
            crystalized_flag=self.if_crystalized_in_circle(self.data[i]['ring1'])
            if self.data[i]['state']>=1:
                self.data[i]['receptive']=0
            elif crystalized_flag==1:
                self.data[i]['receptive']=0
            else:
                self.data[i]['receptive']=1


    def update_equ_1(self):
        return self.current_state+self.gamma

    def update_equ_2(self,ring1_index):
        self.updated_state=0
        for i in ring1_index:
            self.updated_state+=self.data[i]['state']*self.data[i]['receptive']/12
        self.updated_state=self.current_state+self.updated_state+self.gamma
        return self.updated_state

    def update_equ_3(self,ring1_index):
        self.updated_state=0
        for i in ring1_index:
            self.updated_state+=self.data[i]['state']*self.data[i]['receptive']/12
        self.updated_state=self.current_state*0.3+self.updated_state
        return self.updated_state

   

    
    def update_center_3_equ(self,index):
        self.current_state=self.data[index]['state']
        
        #if center larger or equal to 1, remain crystalized
        if self.current_state>=1:
            self.updated_state=self.update_equ_1()
        #if center smaller than 1, not fully crystalized
        else:
            self.crystalized_flag_ring1=self.if_crystalized_in_circle(self.data[index]['ring1'])
            self.crystalized_flag_ring2=self.if_crystalized_in_circle(self.data[index]['ring2'])
            
            #crystalized cell in first circle
            if self.crystalized_flag_ring1==1:
                
                self.updated_state=self.update_equ_2(self.data[index]['ring1'])
            
            #elif crystal in second circle
            elif self.crystalized_flag_ring1==0 and self.crystalized_flag_ring2==1:
                self.updated_state=self.update_equ_3(self.data[index]['ring1'])
            #elif no crystal in 2 circles:
            else:
                self.updated_state=self.update_equ_1()
        return self.updated_state

    def boundary_condition(self,center_pos,this_pos,threshold):
        #if the crystallized cell reaches the threshold, stop the generation
        #0: update it
        dist_to_center=sum((np.array(center_pos)-np.array(this_pos))**2)
        if dist_to_center>threshold**2:
            self.break_while_flag=1
        return self.break_while_flag


    def generation_and_plot(self):
        time_count=0
        while time_count<self.timestep:
            #update the receptive property
            self.decide_receptive()
            
            updated_state_list=[]
            for j in self.data.keys():
                updated_state_list.append(self.update_center_3_equ(j))
            for j in self.data.keys():
                self.data[j]['state']=updated_state_list[j]
                if updated_state_list[j]>=1:
                    self.break_while_flag=self.boundary_condition(self.data[self.center_index]['pos'],self.data[j]['pos'],self.boundary)
            time_count+=1
            
            
            if self.break_while_flag==1:
                break

        plt.style.use('dark_background')
        fig_width = 20
        fig, ax = plt.subplots(figsize=(fig_width, fig_width))

        for i in range(self.N_X*self.N_Y):
            hex = RegularPolygon((self.data[i]['pos'][0], self.data[i]['pos'][1]), numVertices=6, radius=1/np.sqrt(3), 
                                 orientation=np.radians(0),  facecolor = (1,1,1),
                                 alpha=min(1,self.data[i]['state']))#, edgecolor='k')
            ax.add_patch(hex)
        ax.axis('equal')
        # plt.savefig('../image/3equation_T_{}.png'.format(self.T))
        # plt.savefig('../image/3equation_b_{}_g_{}.png'.format(self.beta, self.gamma))

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

    # beta_list=[0.3, 0.5, 0.7]
    # gamma_list=[0.0001, 0.001, 0.01]
    for T in range(-15,1):
    # for beta in beta_list:
    #     for gamma in gamma_list:
            gamma = compute_gamma(T, p_h2o=0.01)
            beta = compute_beta(T, gamma)
            print( gamma, beta)
            generator_3_eqn=three_equation_snowflake_generation(data, N_X, N_Y, beta=beta, gamma=gamma, timestep=2000,T = T)
            generator_3_eqn.generation_and_plot()