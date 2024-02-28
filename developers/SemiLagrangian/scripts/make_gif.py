import matplotlib.pyplot as plt
import imageio
from sys import argv
import numpy as np
from numpy import genfromtxt

# function that will create a single snapshot at time t
def create_frame(x, y , solution, t):
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(left=0 , right=1)
    ax.set_ylim3d(bottom=0 , top=1)
    ax.set_zlim3d(bottom=0 , top=1)
    ax.plot_surface(x, y , solution , cmap='viridis')
    ax.set_title("Transport at {} seconds".format(t))
    plt.savefig(f"img/img_{t}.eps", transparent=False, facecolor="white" , format="eps")
    plt.close()


input_file = str(argv[1])
M = int(argv[2])
K = int(argv[3])
T = float(argv[4])
data = genfromtxt(input_file, delimiter=',')
time_steps = np.arange(0,T,T/K)
x = np.linspace(0,1,M-1)
y = np.linspace(0,1,M-1)

X,Y = np.meshgrid(x,y)
# Create all the frames
for i, t in enumerate(time_steps):
    create_frame(X, Y, data[i].reshape((M-1,M-1)), t)

# Gather all the frames
frames = []
for t in time_steps:
    image = imageio.v2.imread(f'img/img_{t}.eps')
    frames.append(image)
# Now we create the final gif
imageio.mimsave('solution.gif', # output gif
                frames,          # array of input frames
                fps=60)          # optional: frames per second

print("Generated solution.gif")