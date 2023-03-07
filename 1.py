import numpy as np
import matplotlib.pyplot as plt
import igraph
from colour import Color
# let's build a simple graph
a=[
    [1,2],
    [1,3],
    [2,4],
    [4,5],
    [3,5],
    [2,5]
]
g=igraph.Graph.TupleList(a)


def plot(val=None,auto=True):
    label = ['1', '2', '3', '4', '5']
    # red, green, blue, yellow, white
    vals = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0.1, 0.1, 0], [0, 0, 0]]
    #color set 2
    vals = [[0.1, 0, 0], [.1, 0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.5, 0.1]]
    # color set 3 all balck
    vals = [[0.0, 0, 0], [.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    if val is not None:
        for i in range(len(vals)):
            print('Node colors(vals) of i:',vals[i][0])
            print('X(Val) of i:',val[0][i])
            #update red color
            vals[i][0]=np.array(vals[i][0])+(float(val[0][i]))
            #update green color
            vals[i][1]=(np.array(vals[i][1])+(float(val[0][i])))*0.6
            #update blue color
            vals[i][2]=(np.array(vals[i][2])+(float(val[0][i])))*0.6
        print('Updated Node Colors(vals):',vals)
    col = [Color(rgb=(vals[0])), Color(rgb=(vals[1])), Color(rgb=(vals[2])), Color(rgb=(vals[3])), Color(rgb=(vals[4]))]
    colors = [col[0].hex, col[1].hex, col[2].hex, col[3].hex, col[4].hex]
    fig, ax = plt.subplots(1, 1)
    igraph.plot(g,target=ax,vertex_label=label,vertex_color=colors)
    plt.show(block=not(auto))
    plt.pause(0.5)
    plt.close()




# let's begin message passing

# Adjacency matrix
adj_mat=g.get_adjacency()
# we got igraph matrix it won't work with numpy
# so let's convert it to np matrix
adj_mat=np.array(adj_mat.data)

# get degree matrix
D=np.identity(5) # we can assume that this matrix is used in ANN
deg=igraph._indegree(g) # here graph is undirected so indeg or outdeg doesn't matter
D=D*(deg)

# let's calculate inverse of degree matrix
D_inv=np.linalg.inv(D)

# let's multiply inverted degree matrix with adjacency matrix and do the normalization
norm_mat=D_inv@adj_mat

# now we can do the message passing

# message passing
#X*norm_matrix

# Display initial graph
plot(auto=False)


# we have x containing red color intensity
x=np.array([
    # [0.0,0.0,0.0,0,0.9] #set 1 V.high intensity on one node
    # [0.5,0.5,0.5,0.5,0.5] #set 2 equally dividing color intensity
    # [0.0,0.5,0.0,0.0,0.5] # set 3 high intensity on highly connected nodes
    # [0.0,0.9,0.0,0.0,0.9] # set 4 V.high intensity on highly connected nodes
    # [0.9,0.0,0.9,0.9,0.0] # set 5 V.high intensity on end nodes

])
X=x@norm_mat
print(X)
for i in range(5000):
    X=(X@norm_mat)
    # plot(X)
    print('This is our X:',X)
plot(X,auto=False)

'''
What's going on ?
we have 5 nodes in this graph, node one is closest to the red color.
In feature set we have intensity of the red color for each node.
In the set one only 1st node has red color intensity rest of them have 0.

'''