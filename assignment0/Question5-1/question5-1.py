import numpy as np
import matplotlib.pyplot as plt

# Draw 100 samples from a 2D Gaussian distribution with mean [0, 0]
# and an identity covariance matrix.
# Set the mean.
mean = [0, 0]
# Set the identity covariance matrix.
cov = [[1, 0], [0, 1]]
# Generate 100 samples.
samples = np.random.multivariate_normal(mean, cov, 100)
# Plot the samples.
# The plt.figure call creates a new plot.  Everything after that
# is populating and configuring the plot.
plt.figure()
plt.scatter(samples[:, 0], samples[:, 1])
plt.title("2D Gaussian with mean = [0, 0] and identity covariance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid(True)

# How does the scatter polt change if the mean is [1, 1]?
# Set the new mean.
mean = [1, 1]
# Generate 100 samples.
samples2 = np.random.multivariate_normal(mean, cov, 100)
# Plot the samples generated with the shifted mean.
plt.figure()
plt.scatter(samples2[:, 0], samples2[:, 1])
plt.title("2D Gaussian with mean = [1, 1] and identity covariance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid(True)

# Reset the mean to [0, 0] for the remaining questions.
mean = [0, 0]
# How does the scatter plot change if you double the variance?
# Update the covariance matrix.
cov = [[2, 0], [0, 2]]
# Generate 100 samples
samples3 = np.random.multivariate_normal(mean, cov, 100)
# Plot the samples generated with the doubled variance.
plt.figure()
plt.scatter(samples3[:, 0], samples3[:, 1])
plt.title("2D Gaussian with mean = [0, 0] and doubled covariance")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid(True)

# How does the scatter plot change if the covariance is [[1, 0.5], [0.5, 1]]?
# Note: I believe this indicates a correlation of 0.5.
# Update the covariance matrix.
cov= [[1, 0.5], [0.5, 1]]
# Generate 100 samples.
samples4 = np.random.multivariate_normal(mean, cov, 100)
# Plot the samples generated with the correlation 0.5.
plt.figure()
plt.scatter(samples4[:, 0], samples4[:, 1])
plt.title("2D Gaussian with mean = [0, 0] and correlation = 0.5")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid(True)

# How does the scatter plot change if the covariance is [[1, -0.5], [-0.5, 1]]?
# Note: I believe this indicates a correlation of -0.5.
# Update the covariance matrix.
cov = [[1, -0.5], [-0.5, 1]]
# Generate 100 samples.
samples5 = np.random.multivariate_normal(mean, cov, 100)
plt.figure()
plt.scatter(samples5[:, 0], samples5[:, 1])
plt.title("2D Gaussian with mean = [0, 0] and correlation = -0.5")
plt.xlabel("x1")
plt.ylabel("x2")
plt.axis("equal")
plt.grid(True)

# Display all the generated plots.
plt.show()
