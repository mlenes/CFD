# In this python program, we will build a solver for a discretized version of the stokes equation where we have used pressure stabilization.
# So, we will be solving a linear system of the form
#
#		Av + B^T p = f_h
# 		Bv + 1/lambda M p = g_h
#
# where 
#	A is a matrix representing the velocity part, 
#	B is a matrix representing the coupling between velocity and pressure, 
# 	M is a mass matrix, and lambda is a stabilization parameter.
#	We assume that all of these matrices are given and will not be calculated in this program. 

# To solve the system we use the GMRES method, as given by scipy.sparse.linalg
# We will also use a triangular block preconditioner, defined by
# 
# 		P^-1 = [A^-1,	0,
#				0	,	S^-1]
#
# where S = 1/lambda M - B A^-1 B^T
# For the application of the preconditioner, we use the a scipy LinearOperator class to calculate the action of the inverse of P on a vector.
# In particular, we approximate the effect of any inverse matrix on a solver by using a direct solver.

import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator, aslinearoperator, spsolve, inv
from scipy.sparse import csc_matrix, csr_matrix, bmat
from stokes_discretization import assemble_system 

# Solve the Stokes problem using the GMRESSolver
# for a small example

n = 16
nodes = n + 1
lam = 20

A, b = assemble_system(n, lam)  # Example assembly, replace with actual matrices
A = A.tocsr()

# We need to ensure that the rows corresponding to boundaries are set to all zeroes except on the diagonal
# The rows in question are:
# - 0, 1, ..., n (for the first row of the mesh)
# - (n+1) and (2*n) for the second row
# - (2*n+1) and (3*n) for the third row
# - ....
# - (n+1)*n for the last row
#
# Repeat for the second component of the velocity. Which is the same row indices but with
# n**2 added

# Ensure the matrix A is in CSR format for efficient row operations

boundary_indices = [
	list(range(nodes)),                          # bottom: 0 to nodes-1
	list(range(0, nodes**2, nodes)),             # left: 0, nodes, 2*nodes, ...
	list(range(nodes-1, nodes**2, nodes)),       # right: nodes-1, 2*nodes-1, ...
	list(range((nodes-1)*nodes, nodes**2))       # top: (nodes-1)*nodes to nodes**2-1
]

for boundary in boundary_indices:
	for idx in boundary:
		# First velocity component
		A[idx, :] = 0
		A[idx, idx] = 1
		# Second velocity component  
		A[idx + nodes**2, :] = 0
		A[idx + nodes**2, idx + nodes**2] = 1

		# Now me move onto the b vector
		b[idx] = 0
		b[idx + nodes**2] = 0


# Eliminate all set zeroes from the matrix to save space
A.eliminate_zeros()

# Now we can use the GMRESSolver to solve the system
velocity_matrix = A[0 : 2*(n+1)**2 , 0 : 2*(n+1)**2]
coupling_matrix = A[2*(n+1)**2 : , 0 : 2*(n+1)**2]
mass_matrix = A[2*(n+1)**2 : , 2*(n+1)**2 :]	


# Create the preconditioner
def preconditioner(A, B, M, lambda_param):
	A_inv = LinearOperator(A.shape, matvec=lambda x: spsolve(A, x), matmat=lambda X: spsolve(A, X))

	A_inv_B = A_inv.matmat(B.T)
	S_op = M - B @ A_inv_B

	S_inv = LinearOperator(S_op.shape, matvec= lambda x : spsolve(S_op, x))
	
	def apply_preconditioner(v):
		v_u = v[:A.shape[0]]
		v_p = v[A.shape[0]:]
		return np.concatenate([A_inv.matvec(v_u), S_inv.matvec(v_p)])

	return LinearOperator((A.shape[0] + B.shape[0], A.shape[0] + B.shape[0]), matvec=apply_preconditioner)

iterations = 0
def callback(pr_norm):
	"""
	Callback function to print the current iteration number and residual norm.
	"""
	global iterations

	iterations += 1
	if (iterations % 100 == 0):
		print(f"Iteration: {iterations}, Residual Norm: {pr_norm}")

block_matrix = bmat([[velocity_matrix, coupling_matrix.T], 
					 [coupling_matrix, (1 / lam) * mass_matrix]],
					 format='csc')

P_inv = preconditioner(velocity_matrix, coupling_matrix, mass_matrix, lam)
solution , _ = gmres(block_matrix, b, M = P_inv, callback=callback)


u_x = solution[0 : (n+1)**2].reshape((n+1, n+1))
u_y = solution[(n+1)**2 : 2*(n+1)**2].reshape((n+1, n+1))
p = solution[2*(n+1)**2 :]

# plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(u_x, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.colorbar(label='u_x')
plt.title('Velocity Component u_x')
plt.subplot(1, 3, 2)
plt.imshow(u_y, extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.colorbar(label='u_y')
plt.title('Velocity Component u_y')
plt.subplot(1, 3, 3)
plt.imshow(p.reshape((n+1, n+1)), extent=(0, 1, 0, 1), origin='lower', cmap='viridis')
plt.colorbar(label='Pressure p')
plt.title('Pressure p')
plt.tight_layout()
plt.show()