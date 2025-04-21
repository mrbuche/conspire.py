from conspire.constitutive.solid.hyperelastic import NeoHookean
from conspire.fem import Block
import numpy as np


model = NeoHookean(13, 3)
connectivity = np.array([
    [13, 12, 8, 1],
    [10, 3, 0, 8],
    [11, 10, 8, 3],
    [12, 11, 8, 2],
    [11, 2, 3, 8],
    [12, 2, 8, 1],
    [13, 10, 5, 0],
    [13, 11, 10, 8],
    [10, 6, 9, 5],
    [12, 7, 4, 9],
    [12, 11, 7, 9],
    [11, 7, 9, 6],
    [13, 1, 8, 0],
    [13, 9, 4, 5],
    [13, 12, 1, 4],
    [11, 10, 6, 9],
    [11, 10, 3, 6],
    [12, 11, 2, 7],
    [13, 11, 9, 10],
    [13, 12, 4, 9],
    [13, 10, 0, 8],
    [13, 10, 9, 5],
    [13, 12, 11, 8],
    [13, 12, 9, 11],
])
coordinates = np.array([
    [0.5, -0.5, 0.5],
    [0.5, 0.5, 0.5],
    [-0.5, 0.5, 0.5],
    [-0.5, -0.5, 0.5],
    [0.5, 0.5, -0.5],
    [0.5, -0.5, -0.5],
    [-0.5, -0.5, -0.5],
    [-0.5, 0.5, -0.5],
    [0.0, 0.0, 0.5],
    [0.0, 0.0, -0.5],
    [0.0, -0.5, 0.0],
    [-0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.0, 0.0],
])
block = Block(model, connectivity, coordinates)

A = np.zeros((13, 42))
A[0][2] = 1
A[1][5] = 1
A[2][8] = 1
A[3][11] = 1
A[4][26] = 1
A[5][14] = 1
A[6][17] = 1
A[7][20] = 1
A[8][23] = 1
A[9][29] = 1
A[10][18] = 1
A[11][19] = 1
A[12][21] = 1

b = np.zeros((13, 1))
b[0] = 0.8
b[1] = 0.8
b[2] = 0.8
b[3] = 0.8
b[4] = 0.8
b[5] = -0.5
b[6] = -0.5
b[7] = -0.5
b[8] = -0.5
b[9] = -0.5
b[10] = -0.5
b[11] = -0.5
b[12] = -0.5

k = block.nodal_stiffnesses(coordinates)
H = np.zeros((42, 42))
for aa in range(14):
    for bb in range(14):
        for i in range(3):
            for j in range(3):
                H[3 * aa + i][3 * bb + j] = k[aa, bb, i, j]

C = np.zeros((55, 55))
C[:42, :42] = H
C[42:, :42] = A
C[:42, 42:] = A.T
# print(np.linalg.eigvals(H))
# print(np.linalg.eigvals(C))

x = np.zeros((42, 1))
for aa in range(14):
    for i in range(3):
        x[3 * aa + i] = coordinates[aa][i]

forces = block.nodal_forces(coordinates)
f = np.zeros((42, 1))
for aa in range(14):
    for i in range(3):
        f[3 * aa + i] = forces[aa][i]

# residual = np.vstack((f, b - A.dot(x)))
# print(residual)

coords = coordinates * 1.0

residual_norm = 1.0
multipliers = np.zeros((len(b), 1))

while residual_norm > 1e-3:
    forces = block.nodal_forces(coords)
    f = np.zeros((42, 1))
    for aa in range(14):
        for i in range(3):
            f[3 * aa + i] = forces[aa][i]
    residual = np.vstack((f + A.T.dot(multipliers), A.dot(x) - b))
    # check multipliers at right indices?
    # double check constraint indices?
    residual_norm = np.linalg.norm(residual)
    print(residual_norm)
    k = block.nodal_stiffnesses(coordinates)
    H = np.zeros((42, 42))
    for aa in range(14):
        for bb in range(14):
            for i in range(3):
                for j in range(3):
                    H[3 * aa + i][3 * bb + j] = k[aa, bb, i, j]
    C = np.zeros((55, 55))
    C[:42, :42] = H
    C[42:, :42] = A
    C[:42, 42:] = A.T
    sol = np.linalg.inv(C).dot(-residual)
    x += 0.1 * sol[:42]
    multipliers = 0.1 * sol[42:]
    for aa in range(14):
        for i in range(3):
            coords[aa][i] = x[3 * aa + i]