import numpy
import scipy.sparse
import scipy.linalg
from matplotlib import pyplot as plt

import matplotlib.animation as animation

from math import floor, ceil

def generate_mesh(x_min, x_max, n_cells):
	"""Generates a uniform mesh of n_cells on the interval [x_min, x_max].

	Generates a uniform mesh with n_cells on the interval [x_min, x_max].
	A (1D) mesh is a collection of (n_cells + 1) points (vertices) and a 
	list of n_cells pairs of indices of two vertices defining the cells.

	The k-th mesh vertex, x_k, is given by
		x_k = x_min + k delta_x, k = 0, ..., n

	where
		delta_x = (x_max - x_min)/n_cells 

	The k-th cell of the mesh is
		c_k = [k, k + 1]
		
	meaning that cell k is bounded by vertex k and vertex (k+1).

	Parameters
	----------
	x_min : float
		The lower bound of the interval over which the mesh will be generated.
	x_max : float
		The upper bound of the interval over which the mesh will be generated.
	n_cells : int
		The number of cells of the mesh.

	Returns
	-------
	vertices: numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.
	cells: numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell
	"""

	# Make some quantities more clear
	n_vertices = n_cells + 1

	# Generate the vertices
	vertices = numpy.linspace(x_min, x_max, n_vertices)

	# Generate the cells
	cells = numpy.zeros([n_cells, 2], dtype=numpy.int64)
	cells[:, 0] = numpy.arange(0, n_vertices - 1)  # the index of the lower bound vertex of the cell
	cells[:, 1] = numpy.arange(1, n_vertices)  # the index of the lower bound vertex of the cell

	return vertices, cells

def evaluate_local_basis_1D(xi):
	"""Evaluates the two 1D element basis in the local element [0, 1].

	Evaluates the two 1D element basis in the local element [0, 1], at the
	points xi. The points xi must be in the interval [0, 1].

	Since there are two basis the output B_local will be an array such that
		- B_local[0, :] is the evaluation of the first basis over the points xi
		- B_local[1, :] is the evaluation of the second basis over the points xi
		 
	The basis are:
		- B_local_0(x): 1 - x
		- B_local_1(x): x
		
	Parameters
	----------
	xi : numpy.array(float), size [1, n]
		The points in the interval [0, 1] where to evaluate the basis functions.

	Returns
	-------
	B_local : numpy.array(float), size [2, n]
		The two local basis evaluated at the points xi.
	"""

	# Allocate memory for the basis
	B_local = numpy.zeros([2, xi.shape[0]])

	# Compute the basis at the xi points
	B_local[0, :] = 1.0 - xi 
	B_local[1, :] = xi

	return B_local

def gaussian_quadrature(integrand, lower_bound, upper_bound):
	r"""
	Estimate the integral of a funciton using Gaussian quadrature.
	We implemement the method for 2 point gaussian quadrature, which is the approximation
	\int_{a}^{b} f(x) dx ~= (b - a)/2 \sum_{i=1}^{n} w_i f( (b-a)/2 \xi_i + (a+b)/2)
		where specifically for two point Gaussian quadrature
			n = 2
			w_1 = w_2 = 1
			xi_1 = -1/sqrt(3) xi_2 = 1/sqrt(3)
	Parameters
	----------
	integrand : function : np.array(float)  -> np.array(float) 
		The function to integrate.
	lower_bound : float
		The lower bound of the integral.
	upper_bound : float
		The upper bound of the integral.

	Returns
	---------
	integral : float
		The estimated integral of the function.
	"""
	# Gaussian quadrature points and weights
	gauss_points = numpy.array([-1/numpy.sqrt(3), 1/numpy.sqrt(3)])
	gauss_weights = numpy.array([1, 1])

	# Change of variable to map the Gaussian points to the interval [lower_bound, upper_bound]
	gauss_points = (upper_bound - lower_bound) / 2 * gauss_points + (upper_bound + lower_bound) / 2

	# Compute the integral using the Gaussian quadrature formula

	integral = (upper_bound - lower_bound) / 2 * numpy.sum(gauss_weights * integrand(gauss_points))
	return integral

def evaluate_local_basis_1D_d_dx(xi):
	"""Evaluates the derivative of the two 1D element basis in the local element [0, 1].

	Evaluates the derivative of the two 1D element basis in the local element [0, 1], at the
	points xi. The points xi must be in the interval [0, 1].

	Since there are two basis the output B_local_d_dx will be an array such that
		- B_local_d_dx[0, :] is the evaluation of the derivative of the first basis over the points xi
		- B_local_d_dx[1, :] is the evaluation of the derivative of the  second basis over the points xi
		 
	The basis are:
		- B_local_0(x): 1 - x
		- B_local_1(x): x

	Therefore the derivatives are:
		- B_local_0_d_dx(x): -1.0
		- B_local_1_d_dx(x): 1.0
		
	Parameters
	----------
	xi : numpy.array(float), size [1, n]
		The points in the interval [0, 1] where to evaluate the basis functions.

	Returns
	-------
	B_local_d_dx : numpy.array(float), size [2, n]
		The derivative of the two local basis evaluated at the points xi.
	"""

	# Allocate memory for the basis
	B_local_d_dx = numpy.zeros([2, xi.shape[0]])

	# Compute the basis at the xi points
	B_local_d_dx[0, :] = -numpy.ones_like(xi) 
	B_local_d_dx[1, :] = numpy.ones_like(xi)

	return B_local_d_dx

def compute_local_mass_matrix():
	r"""Computes the local mass matrix, for the reference cell [0, 1].

	The local mass matrix M_local is
		M_local[i, j] = \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi
	
	With B_{i}(xi) the local basis function i over the reference cell.

	Since all cells are just an affine rescalling of the reference cell [0, 1],
	a fast way to compute the inner products between all basis is to compute
	first the local inner product on the reference cell and then simply 
	multiply by the required scalling factor due to the coordinate transformation
	to go from the reference cell to the actual cell.
		
	Parameters
	----------
	None

	Returns
	-------
	M_local : numpy.array(float), size [2, 2]
		The local mass matrix (on the reference cell) with M_local[i, j] ~= \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi
	"""

	# The local mass matrix is given by
	# M = | B_1 B_1 & B_1 B_2 |
	#     | B_2 B_1 & B_2 B_2 |

	m1 = lambda xi : evaluate_local_basis_1D(xi)[0, :]**2
	m2 = lambda xi : evaluate_local_basis_1D(xi)[1, :]**2
	m3 = lambda xi : evaluate_local_basis_1D(xi)[0, :] * evaluate_local_basis_1D(xi)[1, :]

	M_local = numpy.zeros([2, 2])
	M_local[0, 0] = gaussian_quadrature(m1, 0.0, 1.0)
	M_local[0, 1] = gaussian_quadrature(m3, 0.0, 1.0)
	M_local[1, 0] = gaussian_quadrature(m3, 0.0, 1.0)
	M_local[1, 1] = gaussian_quadrature(m2, 0.0, 1.0)

	return M_local

def compute_local_stiffness_matrix():
	r"""Computes the local stifness matrix, for the reference cell [0, 1].

	The local stifness matrix M_local is
		M_local[i, j] = \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi
	
	With B_{i}(xi) the local basis function i over the reference cell.

	Since all cells are just an affine rescalling of the reference cell [0, 1],
	a fast way to compute the inner products between all basis is to compute
	first the local inner product on the reference cell and then simply 
	multiply by the required scalling factor due to the coordinate transformation
	to go from the reference cell to the actual cell.
		
	Parameters
	----------
	None

	Returns
	-------
	M_local : numpy.array(float), size [2, 2]
		The local stifness matrix (on the reference cell) with M_local[i, j] ~= \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi
	"""

	# The local mass matrix is given by
	# M = | dB_1/dx dB_1/dx & dB_1/dx dB_2/dx |
	#     | dB_2/dx dB_1/dx & dB_2/dx dB_2/dx |

	m1 = lambda xi : evaluate_local_basis_1D_d_dx(xi)[0, :]**2
	m2 = lambda xi : evaluate_local_basis_1D_d_dx(xi)[1, :]**2
	m3 = lambda xi : evaluate_local_basis_1D_d_dx(xi)[0, :] * evaluate_local_basis_1D_d_dx(xi)[1, :]

	M_local = numpy.zeros([2, 2])
	M_local[0, 0] = gaussian_quadrature(m1, 0.0, 1.0)
	M_local[0, 1] = gaussian_quadrature(m3, 0.0, 1.0)
	M_local[1, 0] = gaussian_quadrature(m3, 0.0, 1.0)
	M_local[1, 1] = gaussian_quadrature(m2, 0.0, 1.0)

	return M_local

def compute_local_convection_matrix():
	r""" compute the local convection matrix, for the reference cell [0,1]
	THis matris is given by
		M_local[i, j] = \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi

	Since all cells are just an affine rescalling of the reference cell [0, 1],
	a fast way to compute the inner products between all basis is to compute
	first the local inner product on the reference cell and then simply 
	multiply by the required scalling factor due to the coordinate transformation
	to go from the reference cell to the actual cell.
		
	Parameters
	----------
	None

	Returns
	-------
	M_local : numpy.array(float), size [2, 2]
		The local convection matrix (on the reference cell) with M_local[i, j] ~= \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi
	"""

	# The local convection matrix is given by
	# M = | B_1 dB_1 & B_1 dB_2 |
	#     | B_2 dB_1 & B_2 dB_2 |

	m1 = lambda xi : evaluate_local_basis_1D(xi)[0, :] * evaluate_local_basis_1D_d_dx(xi)[0, :]
	m2 = lambda xi : evaluate_local_basis_1D(xi)[1, :] * evaluate_local_basis_1D_d_dx(xi)[1, :]
	m3 = lambda xi : evaluate_local_basis_1D(xi)[0, :] * evaluate_local_basis_1D_d_dx(xi)[1, :]
	m4 = lambda xi : evaluate_local_basis_1D(xi)[1, :] * evaluate_local_basis_1D_d_dx(xi)[0, :]

	M_local = numpy.zeros([2, 2])
	M_local[0, 0] = gaussian_quadrature(m1, 0.0, 1.0)
	M_local[0, 1] = gaussian_quadrature(m3, 0.0, 1.0)
	M_local[1, 0] = gaussian_quadrature(m4, 0.0, 1.0)
	M_local[1, 1] = gaussian_quadrature(m2, 0.0, 1.0)

	return M_local

def compute_global_mass_matrix(vertices, cells):
	r"""Computes the global mass matrix, for the mesh of vertices and cells.

	The global mass matrix M_global is
		M_global[i, j] = \int_{\Omega} B_{i}(x) B_{j}(x) dx

	With B_{i}(x) the global basis function i over the domain.

	Parameters
	----------
	vertices : numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.
	cells : numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell

	Returns
	-------
	M_global : numpy.array(float), size [n+1, n+1]
		The global mass matrix (on the whole domain) with M_global[i, j] ~= \int_{\Omega} B_{i}(x) B_{j}(x) dx
	"""

	n_cells = cells.shape[0]
	n_vertices = vertices.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	M_row_idx = numpy.zeros([n_cells, 2, 2])
	M_col_idx = numpy.zeros([n_cells, 2, 2]) 
	M_data = numpy.zeros([n_cells, 2, 2])

	M_local = compute_local_mass_matrix()

	# Note, all integrals scale with h_i where h_i is the size of cell i.
	# This happens due to the fact that the local mass matrix is defined on the reference cell [0,1]

	for cell_idx, cell in enumerate(cells):
		col_idx, row_idx = numpy.meshgrid(cell, cell)
		M_row_idx[cell_idx, :, :] = row_idx
		M_col_idx[cell_idx, :, :] = col_idx
		M_data[cell_idx, :, :] = M_local * delta_x[cell_idx]
	
	M_global = scipy.sparse.csr_array((M_data.flatten(), (M_row_idx.flatten(), M_col_idx.flatten())), shape=(n_vertices, n_vertices))

	return M_global

def compute_global_convection_matrix(vertices, cells):
	r"""Computes the global convection matrix, for the mesh of vertices and cells.

	The global convection matrix M_global is
		M_global[i, j] = \int_{\Omega} B_{i}(x) dB_{j}(x)/dx dx
	
	With B_{i}(x) the global basis function i over the domain.
		
	Parameters
	----------
	vertices : numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.
	cells : numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell

	Returns
	-------
	M_global : numpy.array(float), size [n+1, n+1]
		The global convection matrix (on the whole domain) with M_global[i, j] ~= \int_{\Omega} B_{i}(x) dB_{j}(x)/dx dx
	"""
	
	n_cells = cells.shape[0]
	n_vertices = vertices.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	M_row_idx = numpy.zeros([n_cells, 2, 2])
	M_col_idx = numpy.zeros([n_cells, 2, 2]) 
	M_data = numpy.zeros([n_cells, 2, 2])

	M_local = compute_local_convection_matrix()
	
	# NOTE: for the computation of the global matrix we transform all cells from [0,1] to the size [x_i, x_{i+1}]
	# The calculation of the convection matrix involves the derivative which scales with 1/h_i
	# But the integral scales with h_i so the factors cancel out and we do not need to scale the global matrix
	for cell_idx, cell in enumerate(cells):
		col_idx, row_idx = numpy.meshgrid(cell, cell)
		M_row_idx[cell_idx, :, :] = row_idx
		M_col_idx[cell_idx, :, :] = col_idx
		M_data[cell_idx, :, :] = M_local

	M_global = scipy.sparse.csr_array((M_data.flatten(), (M_row_idx.flatten(), M_col_idx.flatten())), shape=(n_vertices, n_vertices))

	return M_global

def compute_global_stifness_matrix(vertices, cells):
	r"""
	Computes the global stiffness matrix, for the mesh of vertices and cells.
	The global stiffness matrix M_global is
		M_global[i, j] = \int_{\Omega} dB_{i}(x)/dx dB_{j}(x)/dx dx
	With B_{i}(x) the global basis function i over the domain.
	
	Parameters
	----------
	vertices : numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.

	cells : numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell

	Returns
	-------
	M_global : numpy.array(float), size [n+1, n+1]
		The global stiffness matrix (on the whole domain) with M_global[i, j] ~= \int_{\Omega} dB_{i}(x)/dx dB_{j}(x)/dx dx
	"""
	n_cells = cells.shape[0]
	n_vertices = vertices.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	M_row_idx = numpy.zeros([n_cells, 2, 2])
	M_col_idx = numpy.zeros([n_cells, 2, 2]) 
	M_data = numpy.zeros([n_cells, 2, 2])

	M_local = compute_local_stiffness_matrix()
	
	# NOTE: for the computation of the global matrix we transform all cells from [0,1] to the size [x_i, x_{i+1}]
	# The calculation of the stifness matrix involves the two derivatives which scale with 1/h_i
	# But the integral scales with h_i so we need to multiply all terms by 1/h_i
	for cell_idx, cell in enumerate(cells):
		col_idx, row_idx = numpy.meshgrid(cell, cell)
		M_row_idx[cell_idx, :, :] = row_idx
		M_col_idx[cell_idx, :, :] = col_idx
		M_data[cell_idx, :, :] = M_local* delta_x[cell_idx]**(-1)

	M_global = scipy.sparse.csr_array((M_data.flatten(), (M_row_idx.flatten(), M_col_idx.flatten())), shape=(n_vertices, n_vertices))

	return M_global

def compute_SUPG_solution(x_min, x_max, n_cells, phi, tau, u, timestep, boundary, end_time):
	"""Computes the SUPG solution for the convection-diffusion equation.

	The SUPG solution is computed using the global mass matrix and the global convection matrix.
	The solution is computed using the following steps:
	1. Generate the mesh
	2. Compute the global mass matrix
	3. Compute the global convection matrix
	4. Compute the SUPG solution

	Parameters
	----------
	x_min : float
		The lower bound of the interval over which the mesh will be generated.
	x_max : float
		The upper bound of the interval over which the mesh will be generated.
	n_cells : int
		The number of cells of the mesh.
	phi: np.array(float) size [n]
		Array specifying the solution at time 0
	tau : float
		The stabilization parameter.

	u : float
		The velocity of the fluid as specifed per the problem.

	Returns
	-------
	solution : numpy.array(float), size [n+1]
		The solution of the problem at time t + delta t.
	"""

	# Intialize the solution matrix
	solution = numpy.zeros((ceil(end_time/timestep)+1,n_cells+1), dtype =numpy.float64)

	# Specify the initial solution
	solution[0, :] = phi

	# Generate the mesh
	vertices, cells = generate_mesh(x_min, x_max, n_cells)

	# Compute the global mass, convection and stifness matrices
	M_mass_global = compute_global_mass_matrix(vertices, cells)
	M_convection_global = compute_global_convection_matrix(vertices, cells)
	M_stifness_global = compute_global_stifness_matrix(vertices, cells)

	M_convection_global_trans = M_convection_global.transpose()
	
	# Construct the matrix A for the equation
	# 	A phi(t+1) = B phi(t)
	LHS_mat = M_mass_global \
			+ tau * u * M_convection_global_trans \
			+ timestep/2*u*M_convection_global \
			+ timestep/2*tau*(u**2) *M_stifness_global

	# Construct the matrix B for the equation
	# 	A phi(t+1) = B phi(t)
	RHS_mat = M_mass_global \
			+ tau * u * M_convection_global_trans \
			- u*timestep/2*M_convection_global \
			- timestep/2*tau*(u**2) *M_stifness_global

	# Ensure the boundary stays satisfied
	LHS_mat[0, :] = 0.0
	LHS_mat[0, 0] = 1.0
	
	for i in range(1, solution.shape[0]):
		# Construct the right hand side vector
		RHS = RHS_mat@solution[i-1, :]

		# Fix the boundary
		RHS[0] = boundary

		# Compute the solution
		solution[i, :] = scipy.sparse.linalg.spsolve(LHS_mat, RHS)

	return solution, vertices

def assignment1(show = False):
	n_cells = 10
	x_min = 0.0
	x_max = 4.0

	h = (x_max-x_min)/n_cells
	u = 1.0
	tau = u*h/2 * u**(-2)

	# specify the initial solution of 
	# 	phi(x) = (1 +cos(pi * (x-1)))/2 if |x-1| < 1
	#	 and 0 otherwise
	initial_solution = lambda x: (1 + numpy.cos(numpy.pi * (x - 1))) / 2 if 0 <= x <= 2 else 0

	phi_0 = numpy.zeros(n_cells+1)
	
	for i in range(n_cells+1):
		phi_0[i] = initial_solution(x_min + i*h)


	courant_numbers = [0.5, 0.05, 0.005] 
	# plt.figure(figsize=(12, 4))
	for idx, courant_number in enumerate(courant_numbers):
		delta_t = h*courant_number/u
		phi_upwind, vertices = compute_SUPG_solution(x_min, x_max, n_cells, phi = phi_0, tau = 0, u=u, timestep = delta_t, boundary=0, end_time = 1)
		phi_SUPG, vertices = compute_SUPG_solution(x_min, x_max, n_cells, phi = phi_0, tau = tau, u=u, timestep = delta_t, boundary=0, end_time = 1)
	
		n_exact = 1000
		phi_0_exact = numpy.zeros(n_exact+1)
	
		for i in range(n_exact+1):
			phi_0_exact[i] = initial_solution(x_min + i*(x_max-x_min)/n_exact)
	
		phi_exact, vertices_exact = compute_SUPG_solution(x_min, x_max, n_cells = n_exact, phi = phi_0_exact, tau = tau, u=u, timestep = delta_t, boundary=0, end_time = 1)
		print(delta_t, phi_SUPG.shape)
		if show:
			# # Plot the solution at the last time step
			plt.figure(figsize=(8, 4))
			plt.xlabel("x")
			plt.ylabel("phi")
			plt.xlim(x_min, x_max)
			plt.ylim(-0.5, 1.5)
			plt.xticks(vertices)
			plt.grid()

			plt.title(f"Solutions at time {(phi_SUPG.shape[0]-1)*delta_t:.2f} for Courant number {courant_number}")

			# Plot a vertical line at the line x = (phi_supg.shape[0]-1)*delta_t
			plt.axvline(x=1+(phi_SUPG.shape[0]-1)*delta_t, color='gray', linestyle='--', label='Time step')
			
			# Plot the SUPG solution
			plt.plot(vertices, phi_SUPG[-1, :], label="SUPG solution", color="red", linestyle = '-')
	
			# Plot the upwind solution
			plt.plot(vertices, phi_upwind[-1, :], label="Upwind solution", color="blue", linestyle = '-.')
	
			# Plot the exact solution
			plt.plot(vertices_exact, phi_exact[-1, :], label="Exact solution", color="black", linestyle = ":")
	
			plt.legend()
			plt.savefig(f"results/courant_{courant_number}.png")
			plt.show()
			plt.close()
			# def animate(i):
			# 	"""Animation function to plot the solution at each time step."""
			# 	ax.clear()
			# 	ax.set_title("SUPG method")
			# 	ax.set_xlabel("x")
			# 	ax.set_ylabel("phi")
			# 	ax.set_xlim(x_min, x_max)
			# 	ax.set_ylim(-0.5, 1.5)
			# 	ax.set_xticks(vertices)
			# 	ax.set_xticklabels([round(x, 2) for x in vertices])
			# 	ax.grid()
		
			# 	# Plot the SUPG solution
			# 	ax.plot(vertices, phi_SUPG[i, :], label="SUPG solution", color="red", linestyle = '-')
				
			# 	# Plot the upwind solution
			# 	ax.plot(vertices, phi_upwind[i, :], label="Upwind solution", color="blue", linestyle = '-.')
				
			# 	# Plot the exact solution
			# 	ax.plot(vertices_exact, phi_exact[i, :], label="Exact solution", color="black", linestyle = ":")
		
			# 	ax.legend()
			# 	return ax
			
			# # # Create an animation of the upwind and SUPG methods evolving through time
			# fig, ax = plt.subplots(1, 1, figsize=(8, 4))
			
			# # We want the video to be 3 seconds everytime
			# # print(interval, phi_SUPG.shape[0])
			# ani = animation.FuncAnimation(fig, animate, frames=phi_SUPG.shape[0], interval= 1, blit=False, repeat = False)
			
			# plt.show()

		# Make a plot featuring the solutions at the last time step for each courant number
		# This should be a subplot with 3 plots, only show at the end of the for loop
		

	# 	plt.subplot(1, 3, idx+1)
	# 	plt.xlabel("x")
	# 	plt.ylabel("phi")
	# 	plt.xlim(x_min, x_max)
	# 	plt.ylim(-0.5, 1.5)
	# 	plt.xticks(vertices)
	# 	plt.grid()
	# 	# Plot the SUPG solution
	# 	plt.plot(vertices, phi_SUPG[-1, :], label="SUPG solution", color="red", linestyle = '-')
	# 	# Plot the upwind solution
	# 	plt.plot(vertices, phi_upwind[-1, :], label="Upwind solution", color="blue", linestyle = '-.')
	# 	# Plot the exact solution
	# 	plt.plot(vertices_exact, phi_exact[-1, :], label="Exact solution", color="black", linestyle = ":")
	# 	plt.legend()
	# 	plt.title(f"Solutions for Courant number {courant_number}")

	# plt.tight_layout()
	# plt.show()





if __name__ == "__main__":
	N_plot = 100  # resolution for visualization
	import numpy as np
	import matplotlib.pyplot as plt
	from ngsolve import *

	xs = np.linspace(0, 1, N_plot)
	ys = np.linspace(0, 1, N_plot)
	XX, YY = np.meshgrid(xs, ys)
	
	vel_x = np.zeros((N_plot, N_plot))
	vel_y = np.zeros((N_plot, N_plot))
	p_vals = np.zeros((N_plot, N_plot))
	
	for i in range(N_plot):
	    for j in range(N_plot):
	        xx, yy = XX[i, j], YY[i, j]
	        vx, vy = gfu(xx, yy)
	        vel_x[i, j] = vx
	        vel_y[i, j] = vy
	        p_vals[i, j] = gfp(xx, yy)
	
	speed = np.sqrt(vel_x*2 + vel_y*2)
	
	# === Plot Velocity Field ===
	plt.figure(figsize=(7.5, 6))
	plt.streamplot(xs, ys, vel_x, vel_y, color=speed, cmap='viridis', density=2)
	plt.colorbar(label='Velocity magnitude')
	plt.title("Velocity Field (Sampled)")
	plt.xlabel("x")
	plt.ylabel("y")
	plt.axis('tight')
	plt.show()