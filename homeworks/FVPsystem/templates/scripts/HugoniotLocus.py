from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, grid, gca,xlim, title
from numpy import genfromtxt, linspace, exp, sqrt
from sys import argv
from scipy.optimize import fsolve

# Define the argument of the Hugoniot Locus
wstar = 1
vstar = 1

# Define the parameter space
vr = linspace(-3,5,161)

# There are the 1- and 2-locus 
wra = wstar + sqrt(-(vstar-vr)*(-exp(vstar)--exp(vr)))
wrb = wstar - sqrt(-(vstar-vr)*(-exp(vstar)--exp(vr)))

# Plot the 1- and 2-locus
figure()
ax = gca()
color = next(ax._get_lines.prop_cycler)['color']
plot(vr,wra,color=color,label="1-locus")
plot(vr,wrb,color=color,label="2-locus")
xlabel('v')
ylabel('w')
legend()
grid()
savefig("HugoniotLocus.eps")


