from matplotlib.pyplot import figure, plot, savefig, xlabel, ylabel, legend, grid, gca,xlim, title
from numpy import genfromtxt, linspace, exp, sqrt
from sys import argv
from scipy.optimize import fsolve

# First we defined the left-, right-, and intermediate states
uL = [1,1]
ustara = [2.6624, 5.3938]
ustarb = [1.6937, -0.374]
uR = [3,4]

# We compute the shock-speed velocities for all different possible solutions
sLa = (ustara[1]-uL[1])/(uL[0]-ustara[0])
sLb = (ustarb[1]-uL[1])/(uL[0]-ustarb[0])
sRa = (uR[1]-ustara[1])/(ustara[0]-uR[0])
sRb = (uR[1]-ustarb[1])/(ustarb[0]-uR[0])

# We compute the eigenvalues for all possible solutions
lambda1uL = sqrt(exp(1))
lambda1uR = sqrt(exp(3))
lambda2uL = -sqrt(exp(1))
lambda2uR = -sqrt(exp(3))
lambda1ustara = sqrt(exp(ustara[0]))
lambda2ustara = -sqrt(exp(ustara[0]))
lambda1ustarb = sqrt(exp(ustarb[0]))
lambda2ustarb = -sqrt(exp(ustarb[0]))

# We plot only up to t=0.3
tmax = 0.3

# We plot the 1-characteristics for solution a
# We plot multiple parallel lines to show impinging of the characteristics
figure()
for i in range(0,10):
    t = linspace(0,min(tmax,-0.1*i/(sLa-lambda1uL)),101)
    plot((lambda1uL*t-.1*i),t,color="pink")
    t = linspace(0,min(tmax,-.1/(sRa-lambda1uR)*i),101)
    plot((lambda1uR*t+.1*i),t,color="pink")

# We plot the shock speed
t = linspace(0,tmax,101)
plot(sLa*t,t,color="black")
plot(sRa*t,t,color="black")

# We make the figure fancy and save it
xlim([-1,1])
title('1-characteristic solution a')
xlabel('x')
ylabel('t')
savefig("solution_a_lambda1.eps")

# We plot the 2-characteristics for solution a
# We plot multiple parallel lines to show impinging of the characteristics
figure()
for i in range(0,10):
    t = linspace(0,min(tmax,-0.1*i/(sLa-lambda2uL)),101)
    plot((lambda2uL*t-.1*i),t,color="pink")
    t = linspace(0,min(tmax,.1/(sLa-lambda2uR)*i),101)
    plot((lambda2uR*t+.1*i),t,color="pink")

# We plot the shock speed
t = linspace(0,tmax,101)
plot(sLa*t,t,color="black")
plot(sRa*t,t,color="black")

# We make the figure fancy and save it
title('2-characteristic solution a')
xlabel('x')
ylabel('t')
xlim([-1,1])
savefig("solution_a_lambda2.eps")

# We plot the 1-characteristics for solution b
# We plot multiple parallel lines to show impinging of the characteristics
figure()
for i in range(0,10):
    t = linspace(0,min(tmax,-.1*i/(sRb-lambda1uL)),101)
    plot((lambda1uL*t-.1*i),t,color="pink")
    t = linspace(0,tmax,101)
    plot((lambda1uR*t+.1*i),t,color="pink")

# We plot the shock speed
t = linspace(0,tmax,101)
plot(sLb*t,t,color="black")
plot(sRb*t,t,color="black")

# We make the figure fancy and save it
xlim([-1,1])
title('1-characteristic solution b')
xlabel('x')
ylabel('t')
savefig("solution_b_lambda1.eps")

# We plot the 2-characteristics for solution b
# We plot multiple parallel lines to show impinging of the characteristics
figure()
for i in range(0,10):
    t = linspace(0,min(tmax,-0.1*i/(sRb-lambda2uL)),101)
    plot((lambda2uL*t-.1*i),t,color="pink")
    t = linspace(0,min(tmax,.1/(sRb-lambda2uR)*i),101)
    plot((lambda2uR*t+.1*i),t,color="pink")

# We plot the shock speed
t = linspace(0,tmax,101)
plot(sLb*t,t,color="black")
plot(sRb*t,t,color="black")

# We make the figure fancy and save it
title('2-characteristic solution b')
xlabel('x')
ylabel('t')
xlim([-1,1])
savefig("solution_b_lambda2.eps")










