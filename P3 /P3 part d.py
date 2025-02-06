import numpy as np
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
