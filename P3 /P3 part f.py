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
