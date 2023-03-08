import numpy as np
import matplotlib.pyplot as plt
import igraph
from colour import Color
# import cv2

# let's build a simple graph1
g1=[
    [1,2],
    [1,3],
    [2,4],
    [4,5],
    [3,5],
    [2,5]
]

g2=[
    [1,2],
    [1,3],
    [2,4],
    [3,5],
    [2,5]
]

g3=[
    [1,3],
    [2,4],
    [3,5],
    [2,5]
]

#------select graph set---<<<<<<<<<<<<<<<<<<<<<<<<
a=g1
g=igraph.Graph.TupleList(a)

node_colors_red={'1':[],'2':[],'3':[],'4':[],'5':[]}
node_colors_green={'1':[],'2':[],'3':[],'4':[],'5':[]}
node_colors_blue={'1':[],'2':[],'3':[],'4':[],'5':[]}

'''node color set'''
# red, green, blue, yellow, white
val1 = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1], [0.1, 0.1, 0], [0, 0, 0]]
#color set 2
val2 = [[0.1, 0, 0], [.1, 0.5, 0.5], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.5, 0.1]]
# color set 3 all balck
val3 = [[0.0, 0, 0], [.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
# color set 4 one colored and all black
# vals = [[0.4, 0.55, 0.3], [.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

#---------set vals---------<<<<<<<<<<<<<<<<<<
vals=val3


def plot(val=None,auto=True,pause=0.2):
    label = ['1', '2', '3', '4', '5']

    if val is not None:
        for i in range(len(vals)):
            print('Node colors(vals) of i:',vals[i][0])
            print('X(Val) of ',i,' :',val[0][i])
            #update red color
            vals[i][0]=np.array(vals[i][0])+(float(val[0][i]))
            if(vals[i][0]>1): vals[i][0]=1
            #update green color
            vals[i][1]=(np.array(vals[i][1])+(float(val[0][i])))
            if(vals[i][1]>1): vals[i][1]=1
            #update blue color
            vals[i][2]=(np.array(vals[i][2])+(float(val[0][i])))
            if(vals[i][2]>1): vals[i][2]=1
            # store colors to plot changes
            node_colors_red[str(i+1)].append(vals[i][0])
            node_colors_green[str(i+1)].append(vals[i][1])
            node_colors_blue[str(i+1)].append(vals[i][2])
        print('Updated Node Colors(vals):',vals)

    col = [Color(rgb=(vals[0])), Color(rgb=(vals[1])), Color(rgb=(vals[2])), Color(rgb=(vals[3])), Color(rgb=(vals[4]))]
    colors = [col[0].hex, col[1].hex, col[2].hex, col[3].hex, col[4].hex]
    fig, ax = plt.subplots(1, 1)
    igraph.plot(g,target=ax,vertex_label=label,vertex_color=colors)
    plt.show(block=not(auto))
    plt.pause(pause)
    plt.close()




# let's begin message passing

# Adjacency matrix
adj_mat=g.get_adjacency()
# we got igraph matrix it won't work with numpy
# so let's convert it to np matrix
adj_mat=np.array(adj_mat.data)

#add self connections
# self_connect=np.diag([1,1,1,1,1])
# adj_mat=adj_mat+self_connect
# print(adj_mat)
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
    [0.01,0.01,0.01,0.01,0.01] #set 2 equally dividing low color intensity
    # [0.5,0.5,0.5,0.5,0.5] #set 3 equally dividing high color intensity
    # [0.0, 0.01, 0.0, 0.0, 0.01]  # set 4 low intensity on highly connected nodes
    # [0.0,0.5,0.0,0.0,0.5] # set 5 high intensity on highly connected nodes
    # [0.05,0.0,0.05,0.05,0.0] # set 6 high intensity on end nodes

])
X=x@norm_mat
print(X)
for i in range(15):
    X=(X@norm_mat)
    plot(X,pause=0.02)
    print('This is our X:',X)
plot(X,auto=False)

'''
What's going on ?
we have 5 nodes in this graph, node one is closest to the red color.
In feature set we have intensity of the red color for each node.
In the set one only 1st node has red color intensity rest of them have 0.

'''

'''plotting node color values as line'''
print(len(node_colors_red),node_colors_red)
fig,ax=plt.subplots(5)
ax[0].plot(range(len(node_colors_red['1'])),node_colors_red['1'])
ax[0].title.set_text('Red color for node 1')
ax[1].plot(range(len(node_colors_red['2'])),node_colors_red['2'])
ax[1].title.set_text('Red color for node 2')
ax[2].plot(range(len(node_colors_red['3'])),node_colors_red['3'])
ax[2].title.set_text('Red color for node 3')
ax[3].plot(range(len(node_colors_red['4'])),node_colors_red['4'])
ax[3].title.set_text('Red color for node 4')
ax[4].plot(range(len(node_colors_red['5'])),node_colors_red['5'])
ax[4].title.set_text('Red color for node 5')
# plt.legend(['n1','n2','n3','n4','n5'])
plt.show()

# Green color
fig,ax=plt.subplots(5)
ax[0].plot(range(len(node_colors_green['1'])),node_colors_green['1'])
ax[0].title.set_text('Green color for node 1')
ax[1].plot(range(len(node_colors_green['2'])),node_colors_green['2'])
ax[1].title.set_text('Green color for node 2')
ax[2].plot(range(len(node_colors_green['3'])),node_colors_green['3'])
ax[2].title.set_text('Green color for node 3')
ax[3].plot(range(len(node_colors_green['4'])),node_colors_green['4'])
ax[3].title.set_text('Green color for node 4')
ax[4].plot(range(len(node_colors_green['5'])),node_colors_green['5'])
ax[4].title.set_text('Green color for node 5')
plt.show()

# Blue color
fig,ax=plt.subplots(5)
ax[0].plot(range(len(node_colors_blue['1'])),node_colors_blue['1'])
ax[0].title.set_text('Blue color for node 1')
ax[1].plot(range(len(node_colors_blue['2'])),node_colors_blue['2'])
ax[1].title.set_text('Blue color for node 2')
ax[2].plot(range(len(node_colors_blue['3'])),node_colors_blue['3'])
ax[2].title.set_text('Blue color for node 3')
ax[3].plot(range(len(node_colors_blue['4'])),node_colors_blue['4'])
ax[3].title.set_text('Blue color for node 4')
ax[4].plot(range(len(node_colors_blue['5'])),node_colors_blue['5'])
ax[4].title.set_text('Blue color for node 5')
plt.show()


def plot_all_changes(col,name):
    # ax[0].\\
    l1 = plt.plot(range(len(col['1'])), col['1'])
    # ax[0].title.set_text('Red color for node 1')
    # ax[1].\
    l2 = plt.plot(range(len(col['2'])), col['2'])
    # ax[1].title.set_text('Red color for node 2')
    # ax[2].\
    l3 = plt.plot(range(len(col['3'])), col['3'])
    # ax[2].title.set_text('Red color for node 3')
    # ax[3].\
    l4 = plt.plot(range(len(col['4'])), col['4'])
    # ax[3].title.set_text('Red color for node 4')
    # ax[4].\
    l5 = plt.plot(range(len(col['5'])), col['5'])
    # ax[4].title.set_text('Red color for node 5')
    plt.legend(['n1', 'n2', 'n3', 'n4', 'n5'])
    plt.title(str(name+' color change on nodes'))
    plt.show()

plot_all_changes(node_colors_red,'Red')
plot_all_changes(node_colors_green,'Green')
plot_all_changes(node_colors_blue,'Blue')
