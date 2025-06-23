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
from scipy.sparse import csr_matrix

def get_convection_matrix(u_vel_vec):
	"""
	Get convection matrices N1 and N2 for Newton's method on Navier-Stokes equation.
	
	Args:
		u_vel_vec: Velocity vector only (size = V.ndof = 2*(n+1)^2)
		
	Returns:
		N1_scipy: Matrix for N(u^m, u^(m+1)) = (u^m · ∇)u^(m+1)
		N2_scipy: Matrix for N(u^(m+1), u^m) = (u^(m+1) · ∇)u^m
	"""
	from scipy.sparse import csr_matrix
	
	# Calculate amount of meshes from the velocity vector
	n = (len(u_vel_vec) // 2)**0.5 - 1
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
	u_full[:V.ndof] = u_vel_vec  # Fill velocity part, leave pressure as zero
	
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
	N1 = BilinearForm(X)
	
	# Component-wise assembly for N1: (u^m·∇)u^(m+1)
	# Only affects the velocity-velocity block
	for i in range(mesh.dim):  # mesh.dim = 2 for your quad mesh
		for j in range(mesh.dim):
			N1 += u_vel[j] * Grad(u_trial)[i,j] * v_test[i] * dx
	
	N1.Assemble()
	
	# Assemble second convection matrix: N2 = (u^(m+1)·∇)u^m
	# This is (u_trial·∇)u_vel where u_vel is the known iterate u^m
	N2 = BilinearForm(X)
	
	# Component-wise assembly for N2: (u^(m+1)·∇)u^m
	# Only affects the velocity-velocity block
	for i in range(mesh.dim):  # mesh.dim = 2 for your quad mesh
		for j in range(mesh.dim):
			N2 += u_trial[j] * Grad(u_vel)[i,j] * v_test[i] * dx
	
	N2.Assemble()
	
	# Convert both to scipy sparse matrices
	rows1, cols1, vals1 = N1.mat.COO()
	N1_scipy = csr_matrix((vals1, (rows1, cols1)), shape=(N1.mat.height, N1.mat.width))
	N1_scipy.eliminate_zeros()
	
	rows2, cols2, vals2 = N2.mat.COO()
	N2_scipy = csr_matrix((vals2, (rows2, cols2)), shape=(N2.mat.height, N2.mat.width))
	N2_scipy.eliminate_zeros()
	
	return N1_scipy, N2_scipy

def get_convection_rhs_direct(u_vel_vec):
	"""
	Calculate N(u^m, u^m) using the existing convection matrix.
	
	Args:
		u_vel_vec: Velocity vector only (size = V.ndof = 2*(n+1)^2)
		
	Returns:
		rhs_vec: Right-hand side vector N(u^m, u^m)
	"""
	# Get the convection matrix N1 = N(u^m, u^(m+1))
	N1, _ = get_convection_matrix(u_vel_vec)
	

	# Calculate amount of meshes from the velocity vector
	n = (len(u_vel_vec) // 2)**0.5 - 1
	n = int(n)
	
	# Create full vector (velocity + pressure)
	mesh = MakeQuadMesh(n, n)
	V = VectorH1(mesh, order=1, dirichlet=".*")
	Q = H1(mesh, order=1)
	X = V*Q
	V_space = X.components[0]
	
	u_full = np.zeros(X.ndof)
	u_full[:V_space.ndof] = u_vel_vec
	
	# Calculate N(u^m, u^m) = N1 * u_full
	rhs_vec = N1.dot(u_full)
	
	return rhs_vec

def compute_residual(X_stokes, b_stokes, convection_term, u_x, u_y, p):
	"""
	Compute the residual of the Navier-Stokes equation.
	
	Args:
		X: System matrix.
		N: Nonlinear convection term.
		u_x: Velocity component in x-direction.
		u_y: Velocity component in y-direction.
		p: Pressure component.
		
	Returns:
		residual: The computed residual vector.
	"""


	solution = np.concatenate([u_x.flatten(), u_y.flatten(), p.flatten()])
	velocity_solution = np.concatenate([u_x.flatten(), u_y.flatten()])

	# Get the stokes matrix and right hand side
	stokes_residual = b_stokes - X_stokes @ solution

	# Get the convection term
	residual = stokes_residual - convection_term

	return residual

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





	for i in range(10):
		residual_stokes = b_stokes - X_stokes @ np.concatenate([u_x.flatten(), u_y.flatten(), p.flatten()])
		residual_convection = get_convection_rhs_direct(np.concatenate([u_x.flatten(), u_y.flatten()]))
		
		residual = residual_stokes - residual_convection
		print(f"Iteration {i}, residual norm: {np.linalg.norm(residual)}")
		# Extract the velocity component and construct the nonlinear component of Navier Stokes
		velocity = np.concatenate([u_x.flatten(), u_y.flatten()])
		N1, N2 = get_convection_matrix(velocity)
	
		# We now want to solve the linear system with matrix
			# X = [[ A+ N1 + N2, B^T
			#		          B, 1/lambda M]] 
		# and right hand side
		# b = [f_h + N, 0]
	
		# N1 and N2 are just as big as X
		# but only have nonzero values in the upper left bock, so we can safely add everything
		X += N1 + N2 
	
		# N_u_u = (u^m · ∇)u^(m) to the right hand side of the equation
		N_u_u = get_convection_rhs_direct(velocity)
		rhs = b + N_u_u
	
		# Solving the new linear system 
		u_x, u_y, p = stokes_solver(X, rhs, method)

		# residual = compute_residual(X, b_stokes, N_u_u, u_x, u_y, p)

	return u_x, u_y, p

if __name__ == "__main__":
	ux, uy, p = navier_stokes_solver(32, 10e5, method = "ilu")

	from gmres_solver import plot_results
	plot_results(ux, uy, p)


