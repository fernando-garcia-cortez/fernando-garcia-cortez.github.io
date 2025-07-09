# Fernando Garcia Cortez
# Summer 2025

# Phase Portrait for Quintessence w/constant λ

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters 
λ = 2.4      
w_F = 0     # 0 -> matter
            # 1/3 -> radiation

# Fixed points
# These were found using mathematica's Solve[]
points_abc = np.array([
    [0, 0],
    [1, 0],
    [-1, 0]
])
points_de = np.array([
    [λ / np.sqrt(6), np.sqrt(1 - λ**2 / 6)],
    [λ / np.sqrt(6), -np.sqrt(1 - λ**2 / 6)]
])
points_fg = np.array([
    [np.sqrt(3/2) * (1 + w_F) / λ, np.sqrt(3 * (1 - w_F**2) / (2 * λ**2))],
    [np.sqrt(3/2) * (1 + w_F) / λ, -np.sqrt(3 * (1 - w_F**2) / (2 * λ**2))]
])

# System of equations. 
def system(N, vars):
    # N = ln[a]
    x1, x2 = vars
    common = (3/2) * ((1 - w_F) * x1**2 + (1 + w_F) * (1 - x2**2))
    dx1 = -3 * x1 + (np.sqrt(6)/2) * λ * x2**2 + x1 * common
    dx2 = -(np.sqrt(6)/2) * λ * x1 * x2 + x2 * common
    return [dx1, dx2]

# Grid for quiver plot
x1_vals = np.linspace(-1.5, 1.5, 40)
x2_vals = np.linspace(-1.5, 1.5, 40)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Phase portrait vector field
U = np.zeros(X1.shape)
V = np.zeros(X2.shape)

for i in range(X1.shape[0]):
    for j in range(X1.shape[1]):
        dx = system(0, [X1[i, j], X2[i, j]])
        U[i, j] = dx[0]
        V[i, j] = dx[1]

# Normalize arrows for better visualization
magnitude = np.sqrt(U**2 + V**2)
U /= magnitude + 1e-9
V /= magnitude + 1e-9

# Plot phase portrait
fig, ax = plt.subplots(figsize=(12, 12))
ax.tick_params(axis='both', labelsize=20)  # Change tick number font size
plt.quiver(X1, X2, U, V, color='gray', alpha=0.7)

# Add some trajectories given initial conditions
initial_conditions = [(-0.9,0.1),(0,1),(0.5,1),(1,1),(-1,-1),(-1,1),(1,-1),(0.5,0),(0,0.5),(-0.5,0),(1.1,1),(1.2,1),(-1.2,1),(1.2,-0.2)]
list_of_initial_conditions = [list(t) for t in initial_conditions]
array_of_initial_conditions = np.array(list_of_initial_conditions)


for x0 in initial_conditions:
    sol = solve_ivp(system, [0, 10], x0, t_eval=np.linspace(0, 10, 300))
    x1, x2 = sol.y

    # Create a mask: only keep points within [-1.5, 1.5]
    mask = (np.abs(x1) <= 1.5) & (np.abs(x2) <= 1.5)

    # Apply mask to coordinates
    x1_filtered = x1[mask]
    x2_filtered = x2[mask]

    plt.plot(x1_filtered, x2_filtered, color="blue")
    
plt.plot(points_abc[:,0], points_abc[:,1], '*', markersize=16, color='red',label="Points a,b,c")
plt.plot(points_de[:,0], points_de[:,1], 's', markersize=12, color='red',label="Points f,e")
plt.plot(points_fg[:,0], points_fg[:,1], '^', markersize=16, color='red',label="Points f,g")
plt.plot(array_of_initial_conditions[:,0], array_of_initial_conditions[:,1], 'o', markersize=5, color='blue')
plt.xlabel(r'$x_1 \equiv \frac{\kappa \dot{\phi }}{\sqrt{6}H}$', fontsize=20)
plt.ylabel(r'$x_2\equiv \frac{\kappa \sqrt{V}}{\sqrt{3}H}$', fontsize=20)
#plt.title('Phase Portrait XYZ')
plt.grid(True)
plt.axis('equal')
#plt.legend()
plt.show()
plt.savefig("phase_portrait_constant_lambda_2.png", bbox_inches='tight')
