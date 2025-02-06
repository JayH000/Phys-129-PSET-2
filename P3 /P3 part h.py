import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
#PART A 
# Load the data from mesh.dat
filename = "/Users/domholguin/Documents/mesh (1).dat"
points = np.loadtxt(filename, skiprows=1)  

# Compute the convex hull
hull = ConvexHull(points)

# Compute the Delaunay triangulation
tri = Delaunay(points)

# Plot the results
plt.figure(figsize=(8, 6))

# Plot the Delaunay triangulation
plt.triplot(points[:, 0], points[:, 1], tri.simplices, color='gray', alpha=0.6, label="Delaunay Triangulation")

# Plot the convex hull
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'r-', linewidth=2, label="Convex Hull" if 'Convex Hull' not in plt.gca().get_legend_handles_labels()[1] else "")

# Plot the original points
plt.plot(points[:, 0], points[:, 1], 'bo', markersize=4, label="Data Points")

plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Convex Hull and Delaunay Triangulation")
plt.show()


#PART B
#Modify the lifting map to z = f(x, y) = x^2 + xy + y^2
lifted_points = np.hstack((points, (points[:, 0]**2 + points[:, 0]*points[:, 1] + points[:, 1]**2).reshape(-1, 1)))

# Compute the area change for each triangle
area_ratios = []
for simplex in tri.simplices:
    tri_2d = points[simplex]
    tri_3d = lifted_points[simplex]
    # Compute 2D area
    matrix_2d = np.hstack((tri_2d, np.ones((3, 1))))
    area_2d = 0.5 * np.abs(np.linalg.det(matrix_2d))
    # Compute 3D area
    vec1, vec2 = tri_3d[1] - tri_3d[0], tri_3d[2] - tri_3d[0]
    area_3d = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
    # Compute ratio
    area_ratios.append(area_3d / area_2d if area_2d != 0 else 0)
# Convert to numpy array
area_ratios = np.array(area_ratios)

# Plot heatmap
plt.figure(figsize=(8, 6))
triangulation = plt.tripcolor(points[:, 0], points[:, 1], tri.simplices, facecolors=area_ratios, cmap='viridis', edgecolors='k')
plt.colorbar(triangulation, label="Area Ratio")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Area Ratio Heatmap (Before vs. After Lifting)")
plt.show()

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

# Load the data from mesh.dat
filename = "/Users/domholguin/Documents/mesh (1).dat"
points = np.loadtxt(filename, skiprows=1)  

# Lift the points to 3D (map: (x, y) -> (x, y, x^2 + y^2))
lifted_points = np.hstack((points, (points[:, 0]**2 + points[:, 1]**2).reshape(-1, 1)))

# Compute the Delaunay triangulation
tri = Delaunay(points)

# Function to calculate the normal vector of a triangle
def calculate_normal(triangle):
    # Get the 3 vertices of the triangle in 3D
    p1, p2, p3 = triangle

    # Create two edge vectors
    edge1 = p2 - p1
    edge2 = p3 - p1

    # Compute the cross product to get the normal vector
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    return normal

# Prepare to plot the mesh and normals
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh (triangles)
for simplex in tri.simplices:
    # Get the 3 vertices of the triangle in 3D
    triangle = lifted_points[simplex]
    # Plot the triangle edges
    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='gray', alpha=0.5)

    # Calculate the normal vector
    normal = calculate_normal(triangle)
    # Find the centroid of the triangle for normal placement
    centroid = np.mean(triangle, axis=0)
    
    # Plot the normal vector at the centroid
    ax.quiver(centroid[0], centroid[1], centroid[2], normal[0], normal[1], normal[2], length=0.1, color='r')

# Plot the original points
ax.scatter(lifted_points[:, 0], lifted_points[:, 1], lifted_points[:, 2], color='b', s=4, label="Data Points")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Surface Normals of Lifted Mesh")

plt.legend()
plt.show()

# Load the data from mesh.dat
filename = "/Users/domholguin/Documents/mesh (1).dat"
points = np.loadtxt(filename, skiprows=1)  

# Lift the points to 3D (map: (x, y) -> (x, y, x^2 + y^2))
lifted_points = np.hstack((points, (points[:, 0]**2 + points[:, 1]**2).reshape(-1, 1)))

# Compute the Delaunay triangulation
tri = Delaunay(points)

# Function to calculate the normal vector of a triangle
def calculate_normal(triangle):
    # Get the 3 vertices of the triangle in 3D
    p1, p2, p3 = triangle

    # Create two edge vectors
    edge1 = p2 - p1
    edge2 = p3 - p1

    # Compute the cross product to get the normal vector
    normal = np.cross(edge1, edge2)

    # Normalize the normal vector
    normal = normal / np.linalg.norm(normal)

    return normal

# Initialize an array to store the vertex normals
vertex_normals = np.zeros_like(lifted_points)

# Loop through each triangle and compute vertex normals
for simplex in tri.simplices:
    # Get the 3 vertices of the triangle in 3D
    triangle = lifted_points[simplex]
    
    # Calculate the normal for the current triangle
    normal = calculate_normal(triangle)
    
    # Add the normal to each vertex's normal (accumulate normals)
    for i in range(3):
        vertex_normals[simplex[i]] += normal

# Normalize the vertex normals to unit length
for i in range(len(vertex_normals)):
    if np.linalg.norm(vertex_normals[i]) > 0:
        vertex_normals[i] /= np.linalg.norm(vertex_normals[i])

# Prepare to plot the mesh and vertex normals
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh (triangles)
for simplex in tri.simplices:
    # Get the 3 vertices of the triangle in 3D
    triangle = lifted_points[simplex]
    # Plot the triangle edges
    ax.plot_trisurf(triangle[:, 0], triangle[:, 1], triangle[:, 2], color='gray', alpha=0.5)

# Plot the vertex normals
for i, normal in enumerate(vertex_normals):
    ax.quiver(lifted_points[i, 0], lifted_points[i, 1], lifted_points[i, 2],
              normal[0], normal[1], normal[2], length=0.1, color='r', alpha=0.5)

# Plot the original points
ax.scatter(lifted_points[:, 0], lifted_points[:, 1], lifted_points[:, 2], color='b', s=4, label="Data Points")

# Labels and title
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Vertex Normals of Lifted Mesh")

plt.legend()
plt.show()

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