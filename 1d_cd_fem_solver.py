import numpy
import scipy.sparse
import scipy.linalg
from matplotlib import pyplot as plt
from math import sinh, cosh

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

def evaluate_function(u, vertices, cells, n_points_per_cell = 2):
	r"""Evaluates a FEM function with coefficients u on a mesh of vertices and cells.

	Based on the coefficients u and the mesh, compute the FEM function:
		u_eval = \sum_{k=0}^{n} u_{i} B_{i}(x)
	
	The evaluation is made per element. This is not the most efficient way to evaluate
	a FEM function for linear basis. You can just use the coefficients.
		
	Parameters
	----------
	u : numpy.array(float), size [1, n+1]
		The coefficients of the expansion of the function u to evaluate.
	vertices : numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.
	cells : numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell
	n_points_per_cell : [optional] int, default 2
		The number of points on which to evaluate u in each cell. The default (2)
		is sufficient for linear basis. This is added, for the general case.

	Returns
	-------
	u_eval : numpy.array(float), size [n, n_points_per_cell]
		The evaluation of u at each of the n cells over the n_points_per_cell points inside each cell.
	x_eval : numpy.array(float), size [n, n_points_per_cell]
		The global points where u_eval is evaluated. Needed for plotting.
	"""

	# Determine information on the mesh
	n_cells = cells.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	# Plot the basis B_k (combination of the different parts over the cells)
	xi = numpy.linspace(0, 1, n_points_per_cell)  
	u_eval = numpy.zeros([n_cells, n_points_per_cell])  # allocate the space where to store the basis evaluated at each cell
	x_eval = numpy.zeros([n_cells, n_points_per_cell])  # allocate the space where to store the global coordinates of where the basis are evaluated

	B_local_basis = evaluate_local_basis_1D(xi)  # this is the basis evaluated at the point inside each k-element at the points
														#     xi*(delta_x) + vertices[cells[k, 0]]
														# with delta_x the cell size
	
	for cell_idx, cell in enumerate(cells):
		x_eval[cell_idx, :] = xi * delta_x[cell_idx] + vertices[cell[0]]  # convert the local xi coordinates in the cell into global x coordinates
		u_eval[cell_idx, :] = u[cell] @ B_local_basis  # compute the linear combination of basis based on the coefficients
														   # associated to the vertices of the cell
		
	return u_eval, x_eval

def evaluate_function_d_dx(u, vertices, cells, n_points_per_cell = 2):
	r"""Evaluates the derivative of a FEM function with coefficients u on a mesh of vertices and cells.

	Based on the coefficients u and the mesh, compute the FEM function:
		u_eval = \sum_{k=0}^{n} u_{i} dB_{i}(x)/dx
	
	The evaluation is made per element. This is not the most efficient way to evaluate
	the derivative of a FEM function for linear basis. You can just use the coefficients
	and compute differences.
		
	Parameters
	----------
	u : numpy.array(float), size [1, n+1]
		The coefficients of the expansion of the function u to evaluate.
	vertices : numpy.array(float), size [1, n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is
		the index of the vertex, i.e., vertices[k] is the x-coordinate of the
		k-th vertex of the mesh.
	cells : numpy.array(int), size [2, n]
		The indices of the start and end vertex of each cell, i.e.,
		cells[k, 0] is the lower bound vertex of the k-th cell
		cells[k, 1] is the upper bound vertex of the k-th cell
	n_points_per_cell : [optional] int, default 2
		The number of points on which to evaluate u in each cell. The default (2)
		is sufficient for linear basis. This is added, for the general case.

	Returns
	-------
	u_d_dx_eval : numpy.array(float), size [n, n_points_per_cell]
		The evaluation of the derivative of u at each of the n cells over the n_points_per_cell points inside each cell.
	x_eval : numpy.array(float), size [n, n_points_per_cell]
		The global points where u_eval is evaluated. Needed for plotting.
	"""

	# Determine information on the mesh
	n_cells = cells.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	# Plot the basis B_k (combination of the different parts over the cells)
	xi = numpy.linspace(0, 1, n_points_per_cell)  
	u_d_dx_eval = numpy.zeros([n_cells, n_points_per_cell])  # allocate the space where to store the basis evaluated at each cell
	x_eval = numpy.zeros([n_cells, n_points_per_cell])  # allocate the space where to store the global coordinates of where the basis are evaluated

	B_local_basis_d_dx = evaluate_local_basis_1D_d_dx(xi)  # this is the basis evaluated at the point inside each k-element at the points
														#     xi*(delta_x) + vertices[cells[k, 0]]
														# with delta_x the cell size
	
	for cell_idx, cell in enumerate(cells):
		x_eval[cell_idx, :] = xi * delta_x[cell_idx] + vertices[cell[0]]  # convert the local xi coordinates in the cell into global x coordinates
		u_d_dx_eval[cell_idx, :] = u[cell] @ B_local_basis_d_dx / delta_x[cell_idx]  # compute the linear combination of basis based on the coefficients
																					# associated to the vertices of the cell
																					# NOTE: here we need to divide by the cell size
																					# this is because the local basis is evaluated
																					# on the interval [0, 1] but we need it on the interval [x_i, x_{i+1}]
																					# so the derivative needs to be done for the change of variables from xi to x,
																					# this results in the delta_x term added.
		
	return u_d_dx_eval, x_eval

def compute_local_convection_matrix():
	r"""Computes the local convection matrix, for the reference cell [0, 1].

	The local convection matrix M_local is
		M_local[i, j] = \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi
	
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
		The local convection matrix (on the reference cell) with M_local[i, j] ~= \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi
	"""

	# The local convection matrix is given by
	#   M_{local} = <B_{i}, dB_{j}/dx> = \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi, i,j = 0,1
	# we can approximate this with Gauss-Lobatto quadrature by
	#   \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dx dxi ~= 0.5 * B_{i}(xi_{k}) dB_{j}(xi_{k})\dx + 0.5* B_{i}(xi_{k}) dB_{j}(xi_{k})/dx, i,j = 0,1
	# with x_{0} = 0.0, and x_{1} = 1.0. Given the expressions of the basis, we have that
	#   B_{i}(x_{k}) = \delta_{ik}  the Kronecker-delta 
	#								| -1 if j = 0
	# and that dB_{j}(xi_{k})/dx =  |
	#								| 1 if j = 1
	# This allows us to further simplify this expression to
	#   \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dxi dxi ~= | -0.5 if i = 0,1 and j = 0 
	#												 | 0.5 if i = 0.1  and j = 1

	M_local = numpy.zeros([2, 2])
	M_local[0, 0] = -0.5
	M_local[1, 0] = -0.5 
	M_local[0, 1] = 0.5
	M_local[1, 1] = 0.5
	
	return M_local

def compute_local_diffusion_matrix():
	r"""
	Computes the local diffusion matrix, for the reference cell [0, 1].

	The local diffussion matrix N_local is
		N_local[i, j] = \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi
	
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
	N_local : numpy.array(float), size [2, 2]
		The local stiffness matrix (on the reference cell) with N_local[i, j] ~= \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi
	"""

	# The local stiffness matrix is given by
	#   N_{local} = <dB_{i}/dxi, dB_{j}/dxi> = \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi, i,j = 0,1
	# we can exactly compute this
	#                                                    | 1 if i == j
	#  \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi = < 
	#                                                    | -1 if i =/= j
	#

	N_local = numpy.ones([2, 2])
	N_local[0, 1] = -1.0
	N_local[1, 0] = -1.0
	
	return N_local

def compute_global_convection_matrix(vertices, cells):
	r"""Computes the global convectino matrix, for the mesh of vertices and cells.

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
	
	for cell_idx, cell in enumerate(cells):
		col_idx, row_idx = numpy.meshgrid(cell, cell)
		M_row_idx[cell_idx, :, :] = row_idx
		M_col_idx[cell_idx, :, :] = col_idx
		M_data[cell_idx, :, :] = M_local

	M_global = scipy.sparse.csr_array((M_data.flatten(), (M_row_idx.flatten(), M_col_idx.flatten())), shape=(n_vertices, n_vertices))

	return M_global

def compute_global_diffusion_matrix(vertices, cells):
	r"""
	Computes the global stiffness matrix, for the mesh of vertices and cells.

	The global stiffness matrix M_global is
		N_global[i, j] = \int_{\Omega} dB_{i}(x)/dx dB_{j}(x)/dx dx
	
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
	N_global : numpy.array(float), size [n+1, n+1]
		The global stiffness matrix (on the whole domain) with N_global[i, j] ~= \int_{\Omega} dB_{i}(x)/dx dB_{j}(x)/dx dx
	"""

	n_cells = cells.shape[0]
	n_vertices = vertices.shape[0]
	delta_x = numpy.diff(vertices[cells]).flatten()

	N_row_idx = numpy.zeros([n_cells, 2, 2])
	N_col_idx = numpy.zeros([n_cells, 2, 2]) 
	N_data = numpy.zeros([n_cells, 2, 2])

	N_local = compute_local_diffusion_matrix()
	
	for cell_idx, cell in enumerate(cells):
		col_idx, row_idx = numpy.meshgrid(cell, cell)
		N_row_idx[cell_idx, :, :] = row_idx
		N_col_idx[cell_idx, :, :] = col_idx
		# NOTE: N_local contains the inner product between the derivatives, i.e.,
		#    N_local = <dB_{i}/dx, dB_{j}/dx>
		# but, as we have seen, this is computed for the reference cell [0, 1], not
		# the cell we are looping over, which is [x_{i}, x_{j}] and has length delta_x_{i}.
		# Therefore we need to multiply each derivative by the inverse of delta_x_{i} to
		# correctly compute the derivative in the cell. Then, as we saw for the convection matrix,
		# we need to multiply by delta_x_{i} to correctly compute the integral. This gives
		# an overall 1/delta_x_{i} term we need to multiply, as see below.
		N_data[cell_idx, :, :] = N_local / (delta_x[cell_idx])

	N_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

	return N_global

def compute_forcing_term(f, vertices, cells, method, **kwargs):
	r"""
	Computes the forcing term, right hand side, for the mesh of vertices and cells.

	The forcing term F is
		F[j] = \int_{\Omega} f(x) B_{j}(x) dx
	
	With B_{j}(x) the global basis function i over the domain.
		
	Parameters
	----------
	f : func
		The function implementing the right hand side function of the Poisson or Helmholtz equations.
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
	F : numpy.array(float), size [n+1]
		The forcing term with F[j] ~= \int_{\Omega} f(x) B_{j}(x) dx
	
	"""
	if method == "standard":
		n_cells = cells.shape[0]
		n_vertices = vertices.shape[0]
		delta_x = numpy.diff(vertices[cells]).flatten()
	
		F = numpy.zeros(n_vertices)
		for cell_idx, cell in enumerate(cells):
			f_at_cell_vertices = f(vertices[cell])
			F[cell] += 0.5 * f_at_cell_vertices * delta_x[cell_idx]

	elif method == "one point quadrature":
		n_cells = cells.shape[0]
		n_vertices = vertices.shape[0]
		delta_x = numpy.diff(vertices[cells]).flatten()
	
		# Each element of the forcing term is given by
		# F = \frac{\xi_{i-1}}{h_{i-1}} f_{i-1} + ( 1- \frac{\xi_{i}}{h_{i}} ) f_{i}
		# This can be implemented as a matrix multiplication Af. Where A is given as

		# A = \begin{pmatrix} 
		#  1 & 0 & 0 & ... & 0 \\
		# \frac{\xi_{i-1}}{h_{i-1}} & 1- \frac{\xi_{i}}{h_{i}} & 0 & ... & 0 \\
		# 0 & \frac{\xi_{i}}{h_{i}} & 1- \frac{\xi_{i+1}}{h_{i+1}} & ... & 0 \\
		# ...
		# 0 & 0 & 0 & ... & 1 \\
		# \end{pmatrix}

		# Do not worry too much about the first and last row of A. These only compute F at the boundary,
		# and that value for F gets fixed in the compute solution function.
		h = (vertices[-1] - vertices[0])/(n_cells)
		xi = kwargs.get('xi', h/2)

		f = f(vertices + xi)

		A = numpy.zeros((n_vertices, n_vertices))
		A += numpy.diag(numpy.ones(n_vertices)*(1- xi/h))
		A += numpy.diag(numpy.ones(n_vertices-1)*xi/h, -1)

		A[0, 0] = 1.0
		A[-1, -1] = 1.0
		F = A @ f
		F = h*F

	elif method == "SUPG":
		tau = kwargs.get('tau', -1)
		assert tau != -1, "The SUPG method requires a value for tau"

		n_cells = cells.shape[0]
		n_vertices = vertices.shape[0]
		delta_x = numpy.diff(vertices[cells]).flatten()
		h = (vertices[-1] - vertices[0])/(n_cells)
	
		F = numpy.zeros(n_vertices)
		for cell_idx, cell in enumerate(cells):
			f_at_cell_vertices = f(vertices[cell])
			F[cell] += 0.5 * f_at_cell_vertices * delta_x[cell_idx] + tau * 1/h * f_at_cell_vertices
		
		F = h*F
	else:
		raise ValueError(f"Unknown method {method} for the convection-diffusion equation")
	
	return F


def compute_solution(x_min, x_max, n_cells, epsilon, u, f, method):
	r"""
	Computes the solution to the Bubnov-Galerkin approximation for the Convection-Diffusion equation
		d(u\phi)/dx  - d/dx(epsilon d\phi/dx) = f
		phi(0) = 0
		phi(L) = 1

	over the interval [x_min, x_max] with n=n_cells uniformly distributed cells

	Parameters
	------------
	x_min: float
		the lower bound of the interval
	x_max: float
		the upper bound of the interval
	n_cells: int
		the amount of cells to use in the mesh
	epsilon: float
		the diffusion coefficient
	u: float
		the velocity of the flow
	f: func (R^[n_cells] -> R^[n_cells])
		the function implementing the right hand side of the convection diffusion equation
	artificial_diffusion: bool
		a boolean variable that indicates if artificial diffussion should be added to make the solution nodally exact.

	Returns
	----
	vertices: numpy.array(float), size [n+1]
		The x-coordinates of the vertices of the mesh. The index in the array is the index of the vertex, i.e., vertices[k] is the x-coordinate of the k-th vertex of the mesh.
	phi_h : numpy.array(float), size [n+1]
		The numerical solution to the convection-diffusion equation

	"""

	# Generate the mesh
	vertices, cells = generate_mesh(x_min, x_max, n_cells)
	
	if method == "galerkin":
		F = compute_forcing_term(f, vertices, cells, "standard")

	elif method == "one point quadrature":
		h = (x_max-x_min)/(n_cells)
		Peclet = h*(u)/(2*epsilon)  # The peclet number with characteristic length given by h = (x_max-x_min)/(n_cells), remember that n_vertices = n_cells +1
		artificial_diffusion = Peclet*epsilon*( sinh(2*Peclet)/(cosh(2*Peclet)-1) - 1/Peclet)
		epsilon = epsilon + artificial_diffusion

		xi =((artificial_diffusion)/u - h/2)
		F = compute_forcing_term(f, vertices, cells, "one point quadrature", xi = xi)

	elif method == "artificial diffusion":
		h = (x_max-x_min)/(n_cells)
		Peclet = h*(u)/(2*epsilon)  # The peclet number with characteristic length given by h = (x_max-x_min)/(n_cells), remember that n_vertices = n_cells +1
		artificial_diffusion = Peclet*epsilon*( sinh(2*Peclet)/(cosh(2*Peclet)-1) - 1/Peclet)
		epsilon = epsilon + artificial_diffusion

		xi =((artificial_diffusion)/u - h/2)
		F = compute_forcing_term(f, vertices, cells, "standard")

	elif method == "SUPG":
		h = (x_max-x_min)/(n_cells)
		Peclet = h*(u)/(2*epsilon)  # The peclet number with characteristic length given by h = (x_max-x_min)/(n_cells), remember that n_vertices = n_cells +1
		artificial_diffusion = Peclet*epsilon*( sinh(2*Peclet)/(cosh(2*Peclet)-1) - 1/Peclet)
		epsilon = epsilon + artificial_diffusion

		tau = artificial_diffusion/(u**2)

		F = compute_forcing_term(f, vertices, cells, "SUPG", tau = tau)

	else:
		raise ValueError(f"Unknown method {method} for the convection-diffusion equation")

	# Compute global convection matrix
	M_global = compute_global_convection_matrix(vertices, cells)
	M_global.toarray()
	
	# Compute global diffusion matrix
	N_global = compute_global_diffusion_matrix(vertices, cells)
	N_global.toarray()

	A_global = u*M_global+ epsilon*N_global

	# Compute the right hand side
	

	# Include the boundary conditions
	A_global[0, :] = 0.0
	A_global[0, 0] = 1.0
	A_global[-1, :] = 0.0
	A_global[-1, -1] = 1.0
	F[0] = 0 # The boundary condition at x=0 is phi(0) = 0
	F[-1] = 1 # The boundary condition at x=L is phi(L) = 1

	# Due to overlap between the nth and n+1th basis functions do we need to adjust the right hand side of the equation
	# The RHS for the last non-boundary cell is given by
	# \int_{x_n}^{x_n+1} B_n f dx - u\int_{x_n}^{x_n+1} B_n dB_{n+1}/dx dx + \epsilon \int_{x_n}^{x_n+1} dB_{n}/dx dx dB_{n+1}/dx dx  
	# F[-2] += -u/2 + epsilon/(vertices[-1]-vertices[-2])

	A_global = A_global
	# Solve the system
	phi_h = scipy.sparse.linalg.spsolve(A_global, F)

	return vertices, phi_h	

if __name__ == "__main__":
	x_min = 0.0
	x_max = 4.0
	n_cells = 19
	n_points_per_cell = 2  # since the bases are linear, we just need to evaluate them at two points on each cell
	
	u = 1 # The velocity of the flow 

	# Define the right hand side of the equation
	# solution 0 : use f = 0
	# solution = 1 : use f as shown in the assignment
	solution = 1


	if solution == 0:
		f = lambda x : numpy.zeros_like(x) # The right hand side of the equation
	elif solution == 1:
		f = lambda x : numpy.array(list(map( lambda y : (1-y) if y < 1.5 else min(y-2, 0), x)))
	
	plt.figure()

	for idx, epsilon in enumerate([1, 0.1, 0.01]):
		vertices, phi_SUPG = compute_solution(x_min, x_max, n_cells, epsilon, u, f, "SUPG")
		vertices, phi_one_point = compute_solution(x_min, x_max, n_cells, epsilon, u, f, "one point quadrature")
		vertices, phi_galerkin = compute_solution(x_min, x_max, n_cells, epsilon, u, f, "galerkin")
		
		# Short hand for the ratio between the velocity and the diffussion coefficient
		gamma = u/epsilon 
		
		# Define and calculate the exact solution
		if solution == 0:
			vertices_exact = numpy.linspace(x_min, x_max, 1001)
			phi = lambda x : (1 - numpy.exp(gamma*x))/(1-numpy.exp(gamma*x_max))
			phi_exact = phi(vertices_exact)

		elif solution == 1:
			vertices_exact, phi_exact =  compute_solution(x_min, x_max, 1000, epsilon, u, f, "galerkin")
			
		# Plot the numerical solution and the exact solution for comparison
		plt.plot(vertices_exact, phi_exact,'k-', label=r'$\phi_{\mathrm{exact}}$')
		plt.plot(vertices, phi_SUPG, 'b--', label=r'SUPG')
		plt.plot(vertices, phi_one_point, 'r-.', label=r'One point quadrature')
		plt.plot(vertices, phi_galerkin, 'g:', label=r'Galerkin')
		plt.title(f"Convection-Diffusion equation with $\\epsilon = {epsilon:.2f}$")
		plt.legend()	
		plt.savefig(f"results/assignment_1_2_image{3*solution + idx}.png", dpi = 300)
		plt.close()
