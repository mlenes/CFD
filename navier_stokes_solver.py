# In this program we solve the navier stokes equation for incompressible fluids as given by
# 	nu \nabla u + (u \cdot \nabla)u + \nabla p = f_h
#	\nabla \cdot u = 0
#	u = 0 on \partial \Omega

# The forcing term f is constructed such that the solution is given by
# 	u1_exact = x**2*(1-x)**2*(2*y-8*y**3+6*y**5)
#	u2_exact = -y**2*(1-y)**2*(2*x-8*x**3+6*x**5)
# 	p_exact = x*(1-x)

# We will solve the navier stokes equation by using newtons method
# and solving the resulting linear system with our own stokes solver

from stokes_discretization import assemble_system
from gmres_solver import stokes_solver

from ngsolve import *
from ngsolve.meshes import MakeQuadMesh
import pytest
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

def get_convection_matrix(full_vel_vec):
	"""
	Get convection matrix N for Picard iterations for the Navier-Stokes equation.
	
	Args:
		u_vel_vec: Velocity vector only (size = V.ndof = 2*(n+1)^2)
		
	Returns:
		N1_scipy: Matrix for N(u^m, u^(m+1)) = (u^m · ∇)u^(m+1)
		N2_scipy: Matrix for N(u^(m+1), u^m) = (u^(m+1) · ∇)u^m
	"""
	from scipy.sparse import csr_matrix
	
	# Calculate amount of meshes from the velocity vector
	n = (len(full_vel_vec) // 3)**0.5 - 1
	n = int(n)

	# set up the mesh and function spaces.
	# These should match the ones used in assemble_system
	mesh = MakeQuadMesh(n, n)
	V = VectorH1(mesh, order=1, dirichlet=".*")
	Q = H1(mesh, order=1)
	X = V*Q

	# Create full mixed space vector
	u_full = np.zeros(X.ndof)
	V = X.components[0]  # Velocity space
	u_full = full_vel_vec  # Fill velocity part, leave pressure as zero
	
	# Convert to GridFunction
	gfu = GridFunction(X)
	gfu.vec.FV().NumPy()[:] = u_full
	
	# Extract velocity component
	u_vel = gfu.components[0]
	# Get trial and test functions
	u_trial, p_trial = X.TrialFunction()
	v_test, q_test = X.TestFunction()
	
	# Assemble first convection matrix: N1 = (u^m·∇)u^(m+1)
	# This is (u_vel·∇)u_trial where u_vel is the known iterate u^m
	N = BilinearForm(X)
	
	# Component-wise assembly for N1: (u^m·∇)u^(m+1)
	# Only affects the velocity-velocity block
	for i in range(mesh.dim):  # mesh.dim = 2 for your quad mesh
		for j in range(mesh.dim):
			N += u_vel[j] * Grad(u_trial)[i,j] * v_test[i] * dx
	
	N.Assemble()

	matrix = N.mat  # Convert to CSR format for efficiency

	rows, cols, vals = matrix.COO()
	
	N_sparse = coo_matrix((vals, (rows, cols)), shape=(matrix.height, matrix.width))
	N_sparse = N_sparse.tocsr()  # Convert to CSC format for efficient column slicing

	return N_sparse

# def get_convection_rhs_direct(u_vel_vec):
# 	"""
# 	Calculate N(u^m, u^m) using the existing convection matrix.
	
# 	Args:
# 		u_vel_vec: Velocity vector only (size = V.ndof = 2*(n+1)^2)
		
# 	Returns:
# 		rhs_vec: Right-hand side vector N(u^m, u^m)
# 	"""
# 	# Get the convection matrix N1 = N(u^m, u^(m+1))
# 	N1, _ = get_convection_matrix(u_vel_vec)
	

# 	# Calculate amount of meshes from the velocity vector
# 	n = (len(u_vel_vec) // 2)**0.5 - 1
# 	n = int(n)
	
# 	# Create full vector (velocity + pressure)
# 	mesh = MakeQuadMesh(n, n)
# 	V = VectorH1(mesh, order=1, dirichlet=".*")
# 	Q = H1(mesh, order=1)
# 	X = V*Q
# 	V_space = X.components[0]
	
# 	u_full = np.zeros(X.ndof)
# 	u_full[:V_space.ndof] = u_vel_vec
	
# 	# Calculate N(u^m, u^m) = N1 * u_full
# 	rhs_vec = N1.dot(u_full)
	
# 	return rhs_vec

def navier_stokes_solver(n, lam, method = {'direct, ilu'}):
	"""
	Solve the Navier-Stokes equation using Newton's method and a Stokes solver.
	
	Args:
		n: Number of grid points in each direction for the mesh.
		lam: Stabilization parameter.
		method: Method to use for the preconditioner. Can be "direct" or "ilu".
		
	Returns:
		gfu: GridFunction representing the solution (velocity + pressure).
	"""

	# Assemble the system
	X_stokes, b_stokes = assemble_system(n, lam)
	
	# We keep a seperate copy of X_stokes and b_stokes as we need those for residual calculations
	X = X_stokes.copy()
	b = b_stokes.copy()

	# Solve the stokes problem using a custom solver
	u_x, u_y, p = stokes_solver(X, b, method)
	full_solution = np.concatenate([u_x.flatten(), u_y.flatten(), p.flatten()])

	# N_u_m represent the matrix needed for calculating N(u^(m), u^(m+1)) where we solve for u^(m + 1)
	N_u_m = get_convection_matrix(full_solution)

	# Compute the stokes coming from the stokes equation
	residual = b_stokes - X_stokes @ np.concatenate([u_x.flatten(), u_y.flatten(), p.flatten()])

	# Add the residual corresponding to the nonlinear component of Navier Stokes

	residual = residual - N_u_m @ full_solution
	norm_residual = np.linalg.norm(residual)
	print(f"Initial Navier stokes residual norm: {norm_residual:.6e}")
	try:
		while norm_residual > 1e-6:
			print(f"Navier stokes residual norm: {norm_residual:.6e}")
			print()
			# We now want to solve the linear system with matrix
				# X = [[ A+ N, B^T
				#		    B, 1/lambda M]] 
			# and right hand side
			# b = [f_h , 0]
		
			# N1 and N2 are just as big as X
			# but only have nonzero values in the upper left bock, so we can safely add everything
			X = X_stokes + N_u_m
	
			# Solve the stokes problem using a custom solver
			u_x, u_y, p = stokes_solver(X, b, method)
			full_solution = np.concatenate([u_x.flatten(), u_y.flatten(), p.flatten()])
	
			# N_u_m represent the matrix needed for calculating N(u^(m), u^(m+1)) where we solve for u^(m + 1)
			N_u_m = get_convection_matrix(full_solution)
		
			# Compute the stokes coming from the stokes equation
			residual = b_stokes - X_stokes @ full_solution
		
			# Add the residual corresponding to the nonlinear component of Navier Stokes
			residual = residual - N_u_m @ full_solution
			norm_residual = np.linalg.norm(residual)
	
	except KeyboardInterrupt:
		from gmres_solver import plot_results
		print("Solver interrupted, returning current solution.")

		plot_results(u_x, u_y, p)

		# Exit the system
		exit(0)

	return u_x, u_y, p

if __name__ == "__main__":
	ux, uy, p = navier_stokes_solver(32, 10e8, method = "ilu")

	from gmres_solver import plot_results
	plot_results(ux, uy, p)


