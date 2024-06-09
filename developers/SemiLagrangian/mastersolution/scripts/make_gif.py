import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sys import argv
import numpy as np
from numpy import genfromtxt

input_file = str(argv[1])
M = int(argv[2])
K = int(argv[3])
T = float(argv[4])
data = genfromtxt(input_file, delimiter=',')
time_steps = np.arange(0,T,T/K)
x = np.linspace(0,1,M-1)
y = np.linspace(0,1,M-1)

X,Y = np.meshgrid(x,y)

fig = plt.figure()
ax = plt.axes(projection="3d")

def update(i):
    ax.clear()
    ax.set_xlim3d(left=0 , right=1)
    ax.set_ylim3d(bottom=0 , top=1)
    ax.set_zlim3d(bottom=0 , top=1)
    ax.plot_surface(X, Y, data[i].reshape((M-1,M-1)), cmap='viridis')
    ax.set_title("Transport at {} seconds".format(time_steps[i]))

ani = animation.FuncAnimation(fig, update, frames=len(time_steps), interval=1000/60)

ani.save('solution.gif', writer='imagemagick', fps=60)

print("Generated solution.gif")