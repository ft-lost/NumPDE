from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, grid, gca,xlim, title
from numpy import genfromtxt, linspace, exp, sqrt
from sys import argv
from scipy.optimize import fsolve

# Define the arguments of the Hugoniot Loci
vl = 1
wl = 1
vr = 3
wr = 4

# Define the parameter space
vstar = linspace(-3,5,161)

# We have the 1- and 2-locus of both Hugoniot Locus 
wstara = wl + sqrt(-(vl-vstar)*(exp(vstar)-exp(vl)))
wstarb = wl - sqrt(-(vl-vstar)*(exp(vstar)-exp(vl)))
wstarc = wr + sqrt(-(vr-vstar)*(exp(vstar)-exp(vr)))
wstard = wr - sqrt(-(vr-vstar)*(exp(vstar)-exp(vr)))

# Plot the Hugoniot Loci in different colors. Also plot their intersection points.
figure()
ax = gca()
color = next(ax._get_lines.prop_cycler)['color']
plot(vstar,wstara,color=color,label="$u_L$-locus")
plot(vstar,wstarb,color=color,label="_locus")
color = next(ax._get_lines.prop_cycler)['color']
plot(vstar,wstarc,color=color,label="$u_R$-locus")
plot(vstar,wstard,color=color,label="_4-locus")
color = next(ax._get_lines.prop_cycler)['color']
plot([2.6624],[5.3938],'*',color=color,label="(2.6624,5.3938)")
color = next(ax._get_lines.prop_cycler)['color']
plot([1.6937],[-0.374],'*',color=color,label="(1.6937,-0.374)")
xlabel('v')
ylabel('w')
legend()
grid()
savefig("HugoniotLocusIntermediateState.eps")

# Next we numerically find the solution
# There are two points where the Hugoniot Loci cross. The points solve the following sets of equations as defined in f1 and f2
def f1(ustar):
    vstar = ustar[0]
    wstar = ustar[1]

    return [wl + sqrt(-(vl-vstar)*(exp(vstar)-exp(vl))) - wstar, \
            wr + sqrt(-(vr-vstar)*(exp(vstar)-exp(vr))) - wstar]

def f2(ustar):
    vstar = ustar[0]
    wstar = ustar[1]

    return [wl - sqrt(-(vl-vstar)*(exp(vstar)-exp(vl))) - wstar, \
            wr - sqrt(-(vr-vstar)*(exp(vstar)-exp(vr))) - wstar]

# Solve for the intersections
x1 = fsolve(f1,[0,0])
x2 = fsolve(f2,[0,0])

# Print the solution
print(x1)
print(x2)


