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


if __name__=="__main__":
    
    
    N_X = 100
    N_Y = N_X
    X, Y = generateGrid(N_X, N_Y)
    data = genDict(X, Y)
    
    
#%%
    
    # figure settings
    plt.style.use('dark_background')
    fig_width = 20
    fig, ax = plt.subplots(figsize=(fig_width, fig_width))
    
    for i in range(N_X*N_Y):
        hex = RegularPolygon((data[i]['pos'][0], data[i]['pos'][1]), numVertices=6, radius=1/np.sqrt(3), 
                             orientation=np.radians(0),  facecolor = (1,1,1),
                             alpha=0.05, edgecolor='k')
        ax.add_patch(hex)
    ax.axis('equal')


    # visualize specific grid
    i=50 # x index (x from 0 to N_X-1)
    j=50 # y index (y from 0 to N_Y-1)
    
    Visualize(i, j, data, 1)
    Visualize(i, j, data, 0.6, group='ring1')
    Visualize(i, j, data, 0.2, group='ring2')
    
    # # making object with x,y positions (seems useless...)
    # p = GridPoint(data, 40, 30)
    # print(p.pos)
