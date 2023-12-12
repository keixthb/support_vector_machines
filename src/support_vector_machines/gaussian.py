import numpy as np
import matplotlib.pyplot as plt

# Set the seed for reproducibility
np.random.seed(42)

# Number of points
num_points = 10

# Mean and standard deviation for the Gaussian distribution
mean = 1
std_dev = 0.2

# Generate random points from the Gaussian distribution
x_points = np.random.normal(mean, std_dev, num_points)
y_points = np.random.normal(mean, std_dev, num_points)



# Plot the points
plt.scatter(x_points, y_points, color='blue', marker='.', label='using numpy random normal')

# Plot center point
plt.scatter(1, 1, color='red', marker='.', label='Center (1, 1)')

# Set labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('standard deviation = 0.2')

# Add legend
plt.legend()

plt.xlim(0, 2)
plt.ylim(0, 2)

# Show the plot
plt.savefig("gauss.png")
