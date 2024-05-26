from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, grid
from numpy import genfromtxt, linspace
from sys import argv

# Read input and output file from command line
input_file = str(argv[1])
output_file = str(argv[2])

# Read the data
data = genfromtxt(input_file, delimiter=',')
L = len(data)
x = linspace(-2,3,len(data[0]))

# Generate a plot
figure()
for i in range(0,L,40):
    plot(x,data[i],label="t = "+str(i*2./(L-1.)))
xlabel('x')
ylabel('p(x,t)')
legend()
grid()

# save file
savefig(output_file)

print('Generated ' + output_file)