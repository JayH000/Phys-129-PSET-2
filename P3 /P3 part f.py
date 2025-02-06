import numpy as np

# Define the surface as r(x, y) = (x, y, x^2 + y^2)
def surface(x, y):
    return np.array([x, y, x**2 + y**2])

# Compute the first derivatives (tangent vectors)
def first_derivative(x, y):
    r_x = np.array([1, 0, 2*x])  # Partial derivative wrt x
    r_y = np.array([0, 1, 2*y])  # Partial derivative wrt y
    return r_x, r_y

# Compute the second derivatives
def second_derivative(x, y):
    r_xx = np.array([0, 0, 2])  # Second partial derivative wrt x
    r_xy = np.array([0, 0, 0])  # Mixed partial derivative
    r_yy = np.array([0, 0, 2])  # Second partial derivative wrt y
    return r_xx, r_xy, r_yy

# Compute the normal vector
def normal_vector(r_x, r_y):
    normal = np.cross(r_x, r_y)
    return normal / np.linalg.norm(normal)

# Compute the second fundamental form components L, M, N
def second_fundamental_form(x, y):
    r_x, r_y = first_derivative(x, y)
    r_xx, r_xy, r_yy = second_derivative(x, y)
    
    # Compute the normal vector
    normal = normal_vector(r_x, r_y)
    
    # Compute the second fundamental form components
    L = np.dot(normal, r_xx)
    M = np.dot(normal, r_xy)
    N = np.dot(normal, r_yy)
    
    return L, M, N

# Example: Compute the second fundamental form at a point (x, y)
x, y = 1, 1
L, M, N = second_fundamental_form(x, y)

print(f"Second Fundamental Form at ({x}, {y}):")
print(f"L = {L}, M = {M}, N = {N}")
# Output: Second Fundamental Form at (1, 1):
#L = 0.6666666666666666,    M = 0.0, N = 0.6666666666666666

#PART G
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
# Function to compute the first derivatives (tangent vectors)
def first_derivative(x, y):
    r_x = np.array([1, 0, 2*x])  # Partial derivative wrt x
    r_y = np.array([0, 1, 2*y])  # Partial derivative wrt y
    return r_x, r_y

# Function to compute the second derivatives
def second_derivative(x, y):
    r_xx = np.array([0, 0, 2])  # Second partial derivative wrt x
    r_xy = np.array([0, 0, 0])  # Mixed partial derivative
    r_yy = np.array([0, 0, 2])  # Second partial derivative wrt y
    return r_xx, r_xy, r_yy

# Function to compute the normal vector
def normal_vector(r_x, r_y):
    normal = np.cross(r_x, r_y)
    return normal / np.linalg.norm(normal)

# Function to compute the second fundamental form components L, M, N
def second_fundamental_form(x, y):
    r_x, r_y = first_derivative(x, y)
    r_xx, r_xy, r_yy = second_derivative(x, y)
    
    # Compute the normal vector
    normal = normal_vector(r_x, r_y)
    
    # Compute the second fundamental form components
    L = np.dot(normal, r_xx)
    M = np.dot(normal, r_xy)
    N = np.dot(normal, r_yy)
    
    return L, M, N, r_x, r_y, normal

# Function to compute the shape operator (Weingarten map) matrix
def shape_operator(L, M, N, r_x, r_y):
    # Compute the metric tensor (First Fundamental Form)
    E = np.dot(r_x, r_x)
    F = np.dot(r_x, r_y)
    G = np.dot(r_y, r_y)
    
    # Shape operator matrix
    S = np.array([[L/E, M/E], [M/F, N/G]])
    
    return S, E, F, G

# Function to compute the curvatures from the shape operator
def curvatures(S):
    # Diagonalize the shape operator matrix to get the eigenvalues (principal curvatures)
    eigenvalues, _ = np.linalg.eig(S)
    k1, k2 = eigenvalues  # Principal curvatures
    
    # Gaussian curvature
    K = k1 * k2
    
    # Mean curvature
    H = (k1 + k2) / 2
    
    return k1, k2, K, H

# Load the mesh data
filename = "/Users/domholguin/Documents/mesh (1).dat"
points = np.loadtxt(filename, skiprows=1)  

# Compute the Delaunay triangulation
tri = Delaunay(points)

# Initialize lists to store curvature data
principal_curvatures = []
gaussian_curvatures = []
mean_curvatures = []

# Loop over all vertices
for i, (x, y) in enumerate(points):
    # Compute second fundamental form and tangent vectors
    L, M, N, r_x, r_y, normal = second_fundamental_form(x, y)
    
    # Compute the shape operator
    S, E, F, G = shape_operator(L, M, N, r_x, r_y)
    
    # Compute the curvatures from the shape operator
    k1, k2, K, H = curvatures(S)
    
    # Store the results
    principal_curvatures.append((k1, k2))
    gaussian_curvatures.append(K)
    mean_curvatures.append(H)

# Convert the lists to numpy arrays for convenience
principal_curvatures = np.array(principal_curvatures)
gaussian_curvatures = np.array(gaussian_curvatures)
mean_curvatures = np.array(mean_curvatures)

# Visualize the curvatures
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Plot principal curvatures
ax[0].scatter(points[:, 0], points[:, 1], c=principal_curvatures[:, 0], cmap='viridis', label='k1 (max curvature)')
ax[0].scatter(points[:, 0], points[:, 1], c=principal_curvatures[:, 1], cmap='plasma', label='k2 (min curvature)')
ax[0].set_title('Principal Curvatures')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].legend()

# Plot Gaussian curvature
sc1 = ax[1].scatter(points[:, 0], points[:, 1], c=gaussian_curvatures, cmap='coolwarm')
ax[1].set_title('Gaussian Curvature')
ax[1].set_xlabel('X')
ax[1].set_ylabel('Y')
fig.colorbar(sc1, ax=ax[1])

# Plot Mean curvature
sc2 = ax[2].scatter(points[:, 0], points[:, 1], c=mean_curvatures, cmap='inferno')
ax[2].set_title('Mean Curvature')
ax[2].set_xlabel('X')
ax[2].set_ylabel('Y')
fig.colorbar(sc2, ax=ax[2])

plt.tight_layout()
plt.show()