from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, grid
from numpy import genfromtxt, linspace
from sys import argv

# Read input and output file from command line
input_file = str(argv[1])
output_file = str(argv[2])

# Read the data
data = genfromtxt(input_file, delimiter=',')
L = len(data)
x = data[0]
energies = data[1]

# Generate a plot
figure()
plot(x,energies)
xlabel('t')
ylabel('E(u)')
grid()

# Save Figure
savefig(output_file)

print('Generated ' + output_file)