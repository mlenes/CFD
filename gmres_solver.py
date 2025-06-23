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


from stokes_discretization import assemble_system 

def stokes_solver(X, b, method = {"direct, ilu"}):
	import numpy as np
	from scipy.sparse.linalg import gmres, LinearOperator, spsolve, inv
	from scipy.sparse import csc_matrix

	print("starting stokes solver")

	def preconditioner(A, B, M, method = {"direct, ilu"}):
		"""
		Construct a block preconditioner for the stokes problem defined by
	
			P^-1 = [A^-1,	0,
					0	,	S^-1]
	
		where S = M - B @ A^-1 @ B^T
		Args:
			A: Sparse matrix representing the velocity part of the system.
			B: Sparse matrix representing the coupling between velocity and pressure.
			M: Sparse matrix representing the mass matrix.
			method: Method to use for the preconditioner. Can be "direct" or "ilu".
					Direct uses a direct solver to apply A^-1 to a vector
					ILU uses a an incomplete LU decomposition instead
		Returns:
			A LinearOperator that applies the preconditioner.
		"""
	
		if method == "direct":
			# We only need the effect of our inverses on vectors, so we can use linearoperator classes from scipy
			# For A we also need the effect on matrices to calculate the schur complement
			A_inv = LinearOperator(A.shape, matvec=lambda x: spsolve(A, x), matmat=lambda X: spsolve(A, csc_matrix(X) ))
		
			# Calculate the Schur complement S = M - B @ A_inv_B
			A_inv_B = A_inv.matmat(B.T)
			S_op = M - B @ A_inv_B
		
			# Set up the inverse of the Schur complement
			S_inv = LinearOperator(S_op.shape, matvec= lambda x : spsolve(S_op, x))
			
			# We combine our preconditioners into the form
			# 		P^-1 = [A^-1,	0,
			#				0	,	S^-1]
			def apply_preconditioner(v):
				v_u = v[:A.shape[0]]
				v_p = v[A.shape[0]:]
				return np.concatenate([A_inv.matvec(v_u), S_inv.matvec(v_p)])
		
			return LinearOperator((A.shape[0] + B.shape[0], A.shape[0] + B.shape[0]), matvec=apply_preconditioner)
	
		elif method == "ilu":
			from scipy.sparse.linalg import spilu
	
			A_inv = spilu(csc_matrix(A)).solve
	
			# We have to calculate A_inv(B.T) column by column using the solve method
			# Make a sparse matrix like B.T in X
			BT_dense = B.T.toarray()
			A_inv_BT = np.zeros_like(BT_dense)
			for column in range(BT_dense.shape[1]):
				A_inv_BT[:, column] = A_inv(BT_dense[:, column])
	
			# Calculate the schur complement and the effect of its inverse on a vector
			S_op = M - B @ A_inv_BT

			S_inv = spilu(csc_matrix(S_op)).solve
	
			def apply_preconditioner(v):
				v_u = v[:A.shape[0]]
				v_p = v[A.shape[0]:]
				return np.concatenate([A_inv(v_u), S_inv(v_p)])
	
			return LinearOperator((A.shape[0] + B.shape[0], A.shape[0] + B.shape[0]), matvec=apply_preconditioner)
	
	n = (X.shape[0] // 3)**0.5-1  # Calculate n from the size of the matrix X
	n = int(n)
	# Extract the velocity and coupling matrices from the input matrix X
	# The velocity matrix corresponds to the first n^2 rows and columns,
	# the coupling matrix corresponds to the first n^2 rows and the last n^2 columns,
	# and the mass matrix corresponds to the last n^2 rows and columns.
	velocity_matrix = X[0 : 2*(n+1)**2 , 0 : 2*(n+1)**2]
	coupling_matrix = X[2*(n+1)**2 : , 0 : 2*(n+1)**2]
	mass_matrix = X[2*(n+1)**2 : , 2*(n+1)**2 :]	

	# Create the preconditioner
	P_inv = preconditioner(velocity_matrix, coupling_matrix, mass_matrix, method = method)

	# Define the callback function to monitor progress
	iterations = 0

	def callback(pr_norm):
		"""
		Callback function to print the current iteration number and residual norm.
		"""
		nonlocal iterations

		if (iterations % 10 == 0):
			print(f"\t Iteration {iterations}, Residual norm: {pr_norm:.2e}")
	
		iterations += 1
	
	
	solution , _ = gmres(X, b, M = P_inv, callback=callback, restart  = 100)
	
	print()
	print(f"solution converged, total iterations : {iterations}")

	# Split the solutions up into their three components
	u_x = solution[0 : (n+1)**2].reshape((n+1, n+1))
	u_y = solution[(n+1)**2 : 2*(n+1)**2].reshape((n+1, n+1))
	p = solution[2*(n+1)**2 :].reshape((n+1, n+1))

	return u_x, u_y, p

def plot_results(u_x, u_y, p, lam = None, file = None):
	# plot the results
	import matplotlib.pyplot as plt
	
	plt.figure(figsize=(12, 6))
	
	plt.subplot(1, 3, 1)
	# Plot the x velocity
	plt.imshow(u_x, extent=(0, 1, 0, 1), cmap='viridis', origin = 'lower')
	plt.colorbar(label='u_x')
	plt.title('Velocity Component u_x')
	
	# Plot the y velocity
	plt.subplot(1, 3, 2)
	plt.imshow(u_y, extent=(0, 1, 0, 1), cmap='viridis', origin = 'lower')
	plt.colorbar(label='u_y')
	plt.title('Velocity Component u_y')
	
	# Plot the pressure
	plt.subplot(1, 3, 3)
	plt.imshow(p, extent=(0, 1, 0, 1), cmap='viridis', origin='lower')
	plt.colorbar(label='Pressure p')
	
	if lam is not None:
		plt.title(f'Pressure p lambda = {lam:.2f}')
	else:
		plt.title('Pressure p')
	
	plt.tight_layout()

	if file is not None:
		plt.savefig(file, dpi = 400)

	plt.show()


if __name__ == "__main__":
	print()
	# Solve the Stokes problem using the GMRESSolver
	# for a small example
	n = 40
	lam = 10e8
	
	X, b = assemble_system(n, lam)  # Example assembly, replace with actual matrices
	
	# Ensure the matrix A is in csc format for efficient column operations
	X = X.tocsc()

	# u_x, u_y, p = stokes_solver(X, b, method = "direct")
	


	######## Plot smoothed version of the pressure
	# Pass a smoother function over P to remove oscillations
	# import matplotlib.pyplot as plt
	# from scipy.ndimage import gaussian_filter

	# # p = gaussian_filter(p, sigma=1)
	# # plt.figure()
	# # plt.imshow(p, extent=(0, 1, 0, 1), cmap='viridis', origin='lower')
	# # plt.colorbar(label='Pressure p')
	# # plt.title(f'Smoothed Pressure p')
	# # plt.tight_layout()
	# # plt.savefig(f"results/smoothed_pressure_gmres_solver_{n}.png", dpi=400)
	# # plt.show()

	########## Plot the main results
	# Plot the results
	# plot_results(u_x, u_y, p)

	########## Plot the divergence of the velocity field
	# import numpy as np
	# import matplotlib.pyplot as plt
	# divergence = np.gradient(u_x, axis=0) + np.gradient(u_y, axis=1)
	# plt.figure(figsize=(6, 6))
	# plt.imshow(divergence, extent=(0, 1, 0, 1), cmap='viridis', origin='lower') 
	# plt.colorbar(label='Divergence')
	# plt.title('Divergence of Velocity Field')
	# plt.tight_layout()
	# plt.savefig("results/divergence_velocity_field.png", dpi=400)
	# plt.show()

	########## Plot the exact results
	# # We know that the analytic results are given by
	# # u1_exact = x**2*(1-x)**2*(2*y-8*y**3+6*y**5)
	# # u2_exact = -y**2*(1-y)**2*(2*x-8*x**3+6*x**5)
	# # p_exact = x*(1-x)

	# # so lets plot these as well.
	# import numpy as np
	# x = np.linspace(0, 1, n + 1)
	# y = np.linspace(0, 1, n + 1)
	# X, Y = np.meshgrid(x, y)
	# u1_exact = X**2 * (1 - X)**2 * (2 * Y - 8 * Y**3 + 6 * Y**5)
	# u2_exact = -Y**2 * (1 - Y)**2 * (2 * X - 8 * X**3 + 6 * X**5)
	# p_exact = X * (1 - X)
	# # # Plot the exact solutions
	# plot_results(u1_exact, u2_exact, p_exact, file = f"results/gmres_solver_exact.png")

	############# Plot the (manually gathered) convergence results

	# import matplotlib.pyplot as plt
	# import numpy as np

	# nr_cells = np.array([1,2,4,8,16,32,40,50,60,64,70])
	# iters_direct = np.array([0,2,13,15,15,17,19,19,25,21,25])
	# iters_ilu = np.array([0,2,13,15,19,67,75,68,72,76,78])

	# Plot the convergence results
	# plt.figure(figsize=(8, 6))
	# plt.plot(nr_cells, iters_direct, marker='v', label='Direct Solver')
	# plt.plot(nr_cells, iters_ilu, marker='s', label='ILU Solver')
	# plt.xscale('log')
	# plt.yscale('log')
	# plt.xlabel('Number of Cells (n)')
	# plt.ylabel('Number of Iterations')
	# plt.title('Convergence of GMRES Solver')
	# plt.legend()
	# plt.savefig("results/gmres_sovler_convergence.png", dpi=400)
	# plt.show()