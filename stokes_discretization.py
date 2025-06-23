# In this program, we use ngsolve to discretize the stokes equation with pressure stabilization as given by

#  -\nabla^2 u + \nabla p = f_h
#  	\nabla \cdot u + p/lambda = g_h
# 	u = u_\partial on \partial \Omega

# where 
# 	u is the velocity, 
#	p is the pressure, 
#	f_h is the force term, 
#	g_h is the divergence term, 
#	\lambda is a stabilization parameter.
# 	and of course u_\partial is the known velocity on the boundary
# 
# We will use the NGSolve library to set up the finite element spaces 
# and the bilinear form for the Stokes equation.

from ngsolve import *
from ngsolve.meshes import MakeQuadMesh
import numpy as np
from scipy.sparse import coo_matrix


def assemble_system(n, lam):
	"""
	Assemble the linear system corresponding to the discretization of 
	the stokes equation in the form
		Ax = b

	Returns: 
		A sparse matrix in CSR format representing the Stokes system.
		b 
	"""

	mesh = MakeQuadMesh(n, n)
	
	V = VectorH1(mesh, order=1, dirichlet=".*")
	Q = H1(mesh, order=1)
	
	X = V*Q
	
	u, p = X.TrialFunction()
	v, q = X.TestFunction()
	
	stokes = InnerProduct(Grad(u), Grad(v))*dx + div(u)*q*dx - div(v)*p*dx - 1/lam*p*q*dx
	a = BilinearForm(stokes).Assemble()
	
	# Exact solutions
	u1_exact = x**2*(1-x)**2*(2*y-8*y**3+6*y**5)
	u2_exact = -y**2*(1-y)**2*(2*x-8*x**3+6*x**5)
	p_exact = x*(1-x)
	
	# Set boundary conditions
	gfu = GridFunction(X)
	gfu.components[0].Set(CoefficientFunction((u1_exact, u2_exact)), definedon=mesh.Boundaries(".*"))

	# Get the assembled matrix
	matrix = a.mat
	
	rows, cols, vals = matrix.COO()
	
	A = coo_matrix((vals, (rows, cols)), shape=(matrix.height, matrix.width))
	A = A.tocsr()  # Convert to CSC format for efficient column slicing
	
	# Remove zeros to save memory
	A.eliminate_zeros()

	# Start constructing the right hand side
	# Build the exact solutions to get the boundary conditions
	u1_exact = x**2*(1-x)**2*(2*y-8*y**3+6*y**5)
	u2_exact = -y**2*(1-y)**2*(2*x-8*x**3+6*x**5)
	p_exact = x*(1-x)
	
	# Right hand side
	f1 = -(u1_exact.Diff(x, 2) + u1_exact.Diff(y, 2)) + p_exact.Diff(x)
	f2 = -(u2_exact.Diff(x, 2) + u2_exact.Diff(y, 2)) + p_exact.Diff(y)
	# f = CoefficientFunction((f1, f2))

	# f1 = 0
	# f2 = 0
	f = CoefficientFunction((f1, f2))
	
	l = LinearForm(X)
	l += InnerProduct(f, v)*dx
	l.Assemble()

	# Convert to numpy array
	b = l.vec.FV().NumPy()

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
	
	nodes = n + 1
	boundary_indices = [
		list(range(nodes)),                          # bottom: 0 to nodes-1
		list(range(0, nodes**2, nodes)),             # left: 0, nodes, 2*nodes, ...
		list(range(nodes-1, nodes**2, nodes)),       # right: nodes-1, 2*nodes-1, ...
		list(range((nodes-1)*nodes, nodes**2))       # top: (nodes-1)*nodes to nodes**2-1
	]
	
	# We change to lil format as we are changing the sparsity structure
	from scipy.sparse import lil_matrix
	A = lil_matrix(A)
	
	def top_boundary_function(x):
		return 4*x*(1-x)

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


	# for idx in boundary_indices[-1]:
		# x_loc = (idx % nodes) / n  # x-coordinate of the node
		# b[idx] = top_boundary_function(x_loc)

	return A, b

def print_heatmap(sparse_array):
	"""Print a heatmap of a sparse matrix in CSR form."""

	# Convert to a numpy array
	A_numpy = sparse_array.toarray()
	
	# Print a heatmap of A_numpy
	import matplotlib.pyplot as plt
	plt.imshow(A_numpy, cmap='hot', interpolation='nearest')
	plt.colorbar()
	plt.title("Heatmap of the Assembled Stokes Matrix")
	plt.show()


if __name__ == "__main__":
	A, b = assemble_system(4, 1.0)