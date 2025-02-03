import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay

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
#part b 
#lift using map
lifted_points = np.hstack((points, (points[:, 0]**2 + points[:, 1]**2).reshape(-1, 1)))
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
