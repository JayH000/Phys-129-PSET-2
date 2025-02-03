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
