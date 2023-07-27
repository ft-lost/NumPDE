import matplotlib.pyplot as plt

# Data
M = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
Error = [0.618092, 0.154608, 0.0305544, 0.0131097, 0.00636613, 0.00309705, 0.00149105, 0.000694217, 0.000297195, 9.9012e-05]

# Create the plot
plt.figure(figsize=(8, 6))
plt.loglog(M, Error, marker='o', linestyle='-', color='b', label='Error')

# Add labels and title
plt.xlabel('M')
plt.ylabel('Error')
plt.title('Error vs. M')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()