import numpy
import scipy.sparse
import scipy.linalg
from matplotlib import pyplot as plt

def generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y):
    """Generates a uniform mesh of n_cells_x x n_cells_y on the domain [x_min, x_max] x [y_min, y_max].

    Generates a uniform mesh with n_cells_x x n_cells_y on the domain [x_min, x_max] x [y_min, y_max].
    A (2D) mesh is a collection of (n_cells_x + 1) x (n_cells_y + 1) points (vertices) and a 
    list of n_cells_x x n_cells_y pairs of indices of fours vertices defining the cells.

    The vertices of each cell are given in the following order
    
    3          4
     X ------ X
     |        |
     |        |
     X ------ X
    1          2
 
    The vertex (i, j) of the mesh is given by
        V_{i,j} = [x_min + i delta_x, y_min + j delta_y], i = 0, ..., (n_cells_x + 1), j = 0, ..., (n_cells_y + 1)

    and its single index is given by
        V_{k = i + (n_cells_x + 1)*j} = V_{i, j}

    where
        delta_x = (x_max - x_min)/n_cells_x
        delta_y = (y_max - y_min)/n_cells_y 

    The cell (i, j) of the mesh is given by
        c_{i,j} = [V_{i,j}, V_{i+1, j}, V_{i, j+1}, V_{i+1, j+1}]

    and its single index is given by
        c_{k = i + n_cells_x*j} = c_{i, j}.

    Parameters
    ----------
    x_min : float
        The lower bound of the x interval over which the mesh will be generated.
    x_max : float
        The upper bound of the x interval over which the mesh will be generated.
    y_min : float
        The lower bound of the y interval over which the mesh will be generated.
    y_max : float
        The upper bound of the y interval over which the mesh will be generated.   
    n_cells_x : int
        The number of cells of the mesh in the x-direction.
    n_cells_y : int
        The number of cells of the mesh in the y-direction.

    Returns
    -------
    vertices: numpy.array(float), size [(n_cells_x + 1)*(n_cells_x + 1), 2]
        The x- and y-coordinates of the vertices of the mesh. The index in the array is
        the index of the vertex, i.e., vertices[k] is the x- and y-coordinate of the
        k-th vertex of the mesh, with k given as above V_{k = i + (n_cells_x + 1)*j} = V_{i, j}.
    cells: numpy.array(int), size [n_cells_x * n_cells_x, 4]
        The indices of the four vertices of each cell of the mesh in the order specified above, i.e.,
        cells[k, 0] is the lower left vertex of the k-th cell
        cells[k, 1] is the lower right vertex of the k-th cell
        cells[k, 2] is the upper left vertex of the k-th cell
        cells[k, 3] is the upper right vertex of the k-th cell
        Again, recall the relation between the linear indexing k and the tuple indexing (i,j),
            k = i + n_cells_x * j
    """

    # Make some quantities more clear
    n_vertices_x = n_cells_x + 1
    n_vertices_y = n_cells_y + 1
    n_cells = n_cells_x * n_cells_y

    # Generate the vertices
    x, y = numpy.meshgrid(numpy.linspace(x_min, x_max, n_vertices_x), numpy.linspace(y_min, y_max, n_vertices_y))
    vertices = numpy.hstack([x.reshape(-1, 1), y.reshape(-1, 1)])

    # Generate the cells
    cells = numpy.zeros([n_cells, 4], dtype=numpy.int64)
    for j in range(0, n_cells_y):
        for i in range(0, n_cells_x):
            k = i + n_cells_x*j  # the linear cell number
            
            # Now we add the linear index of the vertices of the cell
            # Recall that the vertices of each cell are given in the following order
            #
            #  3          4
            #   X ------ X
            #   |        |
            #   |        |
            #   X ------ X
            #  1          2
            #
            # The vertex (i, j) of the mesh is given by
            # V_{i,j} = [x_min + i delta_x, y_min + j delta_y], i = 0, ..., (n_cells_x + 1), j = 0, ..., (n_cells_y + 1)
            #
            # The cell k = i + n_cells_x(j - 1) of the mesh is given by
            #   c[k, :] = [V_{i,j}, V_{i+1, j}, V_{i, j+1}, V_{i+1, j+1}]
            cells[k, 0] = (i) + (n_cells_x + 1)*(j)  # the linear index of the lower left corner of the element
            cells[k, 1] = (i+1) + (n_cells_x + 1)*(j)  # the linear index of the lower right corner of the element
            cells[k, 2] = (i) + (n_cells_x + 1)*(j+1)  # the linear index of the upper left corner of the element
            cells[k, 3] = (i+1) + (n_cells_x + 1)*(j+1)  # the linear index of the upper right corner of the element

    return vertices, cells

def compute_local_mass_matrix_1D():
    r""" Computes the local mass matrix for the reference cell [0,1]

    The local mass matrix L_matrix is
        L_local[i,j] = \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi

    Since all cells are just an affine rescalling of the reference cell [0, 1],
    a fast way to compute the inner products between all basis is to compute
    first the local inner product on the reference cell and then simply 
    multiply by the required scalling factor due to the coordinate transformation
    to go from the reference cell to the actual cell.

    This matrix is utilized to compute the 2d local convection matrix.

    Parameters
    ----
    None

    Returns
    ----
    L_local : numpy.array(float), size [2,2]
        The local mass matrix (on the reference cell) with L_local = \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi
    """

    # The local mass matrix is given by
    #   L_{local} = <B_{i}, B_{j}> = \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi, i,j = 0,1
    # we can approximate this with Gauss-Lobatto quadrature by
    #   \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi ~= 0.5 * B_{i}(xi_{k})B_{j}(xi_{k}) + 0.5* B_{i}(xi_{k})B_{j}(xi_{k}), i,j = 0,1
    # with x_{0} = 0.0, and x_{1} = 1.0. Given the expressions of the basis, we have that
    #   B_{i}(x_{k}) = \delta_{ik}  the Kronecker-delta
    # This allows us to further simplify this expression to
    #   \int_{0}^{1} B_{i}(xi) B_{j}(xi) dxi ~= 0.5

    L_local = numpy.zeros([2, 2])
    L_local[0, 0] = 0.5
    L_local[1, 1] = 0.5
    
    return L_local

def compute_local_convection_matrix_1D():
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
    #                               | -1 if j = 0
    # and that dB_{j}(xi_{k})/dx =  |
    #                               | 1 if j = 1
    # This allows us to further simplify this expression to
    #   \int_{0}^{1} B_{i}(xi) dB_{j}(xi)/dxi dxi ~= | -0.5 if i = 0,1 and j = 0 
    #                                                | 0.5 if i = 0.1  and j = 1

    M_local = numpy.zeros([2, 2])
    M_local[0, 0] = -0.5
    M_local[1, 0] = -0.5 
    M_local[0, 1] = 0.5
    M_local[1, 1] = 0.5
    
    return M_local

def compute_local_diffusion_matrix_1D():
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
        The local diffusion matrix (on the reference cell) with N_local[i, j] ~= \int_{0}^{1} dB_{i}(xi)/dxi dB_{j}(xi)/dxi dxi
    """

    # The local diffusion matrix is given by
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

def compute_local_convection_matrix_2D():
    r""" Computes the 2d local convection matrix for the reference cell [0,1] x [0,1]
    The local convection matrix is
        M_local[i, j] = \int_{0}^{1} \int_{0}^{1} \nabla*B_{i}(xi, eta) *B_{j}(xi, eta) dxi deta

    As before, since all cells are just an affine rescalling of the reference cell [0, 1] x [0, 1],
    a fast way to compute the inner products between all basis is to compute
    first the local inner product on the reference cell and then simply 
    multiply by the required scalling factor due to the coordinate transformation
    to go from the reference cell to the actual cell.

      Another improvement is to note that the basis in 2D are just the tensor product of
    1D basis:
        B_{k}(xi, eta) = B_{i}(xi) B_{j}(eta)
    with the linear index k given by:
        k = i + 2*j

    This means that:
        M_local[k, l] = - \int_{0}^{1} B_{r}(xi) dB_{i}(xi)/dxi dxi \int_{0}^{1} B_{j}(eta) B_{s}(eta)/eta deta
                        - \int_{0}^{1} B_{r}(xi) B_{i}(xi) dxi \int_{0}^{1} dB_{j}(eta)/deta B_{s}(eta)/eta deta 
    with the linear indices k and l given by:
        k = i + 2*j
        l = r + 2*s

    We can go one step further and recall that the 1D M_local_1D matrix entries are just the 1D integrals we see
    in the expression above, therefore we can write:
        M_local[k, l] = -M_local_1D[r, i] * L_local_1D[s, j] - L_Local_1D[r, i] * M_local_1D[s, j]
    where L_local_1D refers to the 1D mass matrix

    Note that now our M_local is a 4 x 4 matrix, since there are 4 basis per element (one per vertex).

    We need to scale the terms in the two above expressions differently. Hence, they are returned seperately.

    The local co
    Parameters
    ----------
    None

    Returns
    -------
    M_local : numpy.array(float), size [4, 4]
        The local mass matrix (on the reference cell) with 
        M_local[0, k, l] ~= \int_{0}^{1} \int_{0}^{1} B_{l}(xi, eta) dB_{k}(xi, eta)/dxi dxi deta
        M_local[1, k, l] ~= \int_{0}^{1} \int_{0}^{1} B_{l}(xi, eta) dB_{k}(xi, eta)/deta dxi deta

    """
    # Following the alogirithm described above, we compute the 1D local mass and convection matrices
    M_local_1D = compute_local_convection_matrix_1D()
    L_local_1D = compute_local_mass_matrix_1D()

    # Populate the 2D M_local matrix
        # Populate the 2D M_local matrix
    M_local = numpy.zeros([2, 4, 4])
    for i in range(0, 2):
        for j in range(0, 2):
            for r in range(0, 2):
                for s in range(0, 2):
                    # Compute the linear indices of the basis
                    k = i + 2*j
                    l = r + 2*s

                    # Compute the 2D inner product using a tensor product of the 1D ones
                    M_local[0, k, l] = -M_local_1D[r, i] * L_local_1D[s, j]
                    M_local[1, k, l] = -L_local_1D[r, i] * M_local_1D[s, j]

    return M_local

def compute_local_diffusion_matrix_2D():
    r"""Computes the two components of the local the diffusion matrix, for the 2D reference cell [0, 1] x [0, 1].

    The local diffusion matrix N_local is
        N_local[k, l] = \int_{0}^{1} \int_{0}^{1} \nabla B_{k}(xi, eta) \cdot \nabla B_{l}(xi, eta) dxi deta
    
    With B_{k}(xi, eta) the local %%bashis function k over the 2D reference cell.

    As before, since all cells are just an affine rescalling of the reference cell [0, 1] x [0, 1],
    a fast way to compute the inner products between all basis is to compute
    first the local inner product on the reference cell and then simply 
    multiply by the required scalling factor due to the coordinate transformation
    to go from the reference cell to the actual cell.

    Another improvement is to note that the basis in 2D are just the tensor product of
    1D basis:
        B_{k}(xi, eta) = B_{i}(xi) B_{j}(eta)
    with the linear index k given by:
        k = i + 2*j

    This means that:
        N_local[k, l] = \int_{0}^{1} dB_{i}(xi)/dxi dB_{r}(xi)/dxi dxi *  \int_{0}^{1} B_{j}(eta) B_{s}(eta) deta +
                      + \int_{0}^{1} B_{i}(xi) B_{r}(xi) dxi * \int_{0}^{1} dB_{j}(eta)/deta dB_{s}(eta)/deta deta
    with the linear indices k and l given by:
        k = i + 2*j
        l = r + 2*s

    We can go one step further and recall that the 1D M_local_1D and the 1D N_local_1D matrix entries are just 
    the 1D integrals we see in the expression above, therefore we can write:
        N_local[k, l] = N_local_1D[i, r] * M_local_1D[j, s] + M_local_1D[i, r] * N_local_1D[j, s]
    
    Note that now our M_local is a 4 x 4 matrix, since there are 4 basis per element (one per vertex).

    Each of the terms that are summed on the right hand side of the last expression need to be scalled in 
    different ways, hence we return both terms separately.
    
    Parameters
    ----------
    None

    Returns
    -------
    N_local : numpy.array(float), size [2, 4, 4]
        The two components of the local diffusion matrix (on the reference cell) with 
        N_local[0, k, l] ~= \int_{0}^{1} \int_{0}^{1} dB_{k}(xi, eta)/dxi \nabla dB_{l}(xi, eta)/dxi dxi deta
        N_local[1, k, l] ~= \int_{0}^{1} \int_{0}^{1} dB_{k}(xi, eta)/deta \nabla dB_{l}(xi, eta)/deta dxi deta
    """

    # Following the algorithm described above, we first compute the local 1D mass and diffusion matrices
    M_local_1D = compute_local_mass_matrix_1D()
    N_local_1D = compute_local_diffusion_matrix_1D()

    # Populate the 2D M_local matrix
    N_local = numpy.zeros([2, 4, 4])
    for i in range(0, 2):
        for j in range(0, 2):
            for r in range(0, 2):
                for s in range(0, 2):
                    # Compute the linear indices of the basis
                    k = i + 2*j
                    l = r + 2*s

                    # Compute the 2D inner product using a tensor product of the 1D ones
                    N_local[0, k, l] = N_local_1D[i, r] * M_local_1D[j, s]
                    N_local[1, k, l] = M_local_1D[i, r] * N_local_1D[j, s]
    
    return N_local

def compute_global_convection_matrix_2D(vertices, cells, u):
    r"""Computes the global convection matrix, for the 2D mesh of vertices and cells.

    The global convection matrix M_global is
        M_local[i, j] = \int_{\Omega} \nabla*B_{i}(x, y) *B_{j}(x, y) dx dy

    With B_{i}(x, y) the global basis function i over the domain.
        
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

    u : numpy array(float), size [2]
        The velocity vector field. u[0] is in the x direction and u[1] is in the y direction

    Returns
    -------
    M_global : numpy.array(float), size [n+1, n+1]
        The global convection matrix (on the whole domain) with
        M_local[i, j] = \int_{\Omega} \nabla*B_{i}(x, y) *B_{j}(x, y) dx dy
    """

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    M_row_idx = numpy.zeros([n_cells, 4, 4])
    M_col_idx = numpy.zeros([n_cells, 4, 4]) 
    M_data = numpy.zeros([n_cells, 4, 4])

    M_local = compute_local_convection_matrix_2D()

    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        M_row_idx[cell_idx, :, :] = row_idx
        M_col_idx[cell_idx, :, :] = col_idx

        # NOTE: N_local contains the inner product between the B_k and its derivative, i.e.,
        #    M_local = < u B_{k}, \nabla B_{l}>
        # but, as we have seen, this is computed for the reference cell [0, 1] x [0, 1], not
        # the cell we are looping over, which is [x_{i}, x_{j}] x [y_{r}, y_{s}] and has 
        # lengths delta_x_{k} and delta_y_{l}.
        # Therefore we need to multiply each derivative by scalling coefficients to
        # correctly compute the derivative in the cell. Then, as we saw for the mass matrix,
        # we need to multiply by delta_x_{i} to correctly compute the integral.
        # The above operations cancel out and we are left with only either a delta_y term or a delta_x term.

        M_data[cell_idx, :, :] = u[0]*M_local[0]*delta_y[cell_idx] + u[1]*M_local[1]*delta_x[cell_idx]

    M_global = scipy.sparse.csr_array((M_data.flatten(), (M_row_idx.flatten(), M_col_idx.flatten())), shape=(n_vertices, n_vertices))

    return M_global

def compute_global_diffusion_matrix_2D(vertices, cells):
    r"""Computes the global diffusion matrix, for the 2D mesh of vertices and cells.

    The global diffusion matrix N_global is
        N_global[i, j] = \int_{\Omega} \nabla B_{i}(x, y) \cdot \nabla B_{j}(x, y) dxdy
    
    With B_{i}(x, y) the global basis function i over the domain.
        
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
        The global diffusion matrix (on the whole domain) with
        N_global[i, j] ~= \int_{\Omega} \nabla B_{i}(x, y) \cdot \nabla B_{j}(x, y) dxdy
    """

    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()

    N_row_idx = numpy.zeros([n_cells, 4, 4])
    N_col_idx = numpy.zeros([n_cells, 4, 4]) 
    N_data = numpy.zeros([n_cells, 4, 4])

    N_local = compute_local_diffusion_matrix_2D()
    
    for cell_idx, cell in enumerate(cells):
        col_idx, row_idx = numpy.meshgrid(cell, cell)
        N_row_idx[cell_idx, :, :] = row_idx
        N_col_idx[cell_idx, :, :] = col_idx
        # NOTE: N_local contains the inner product between the derivatives, i.e.,
        #    N_local = <\nabla B_{k}, \nabla B_{l}>
        # but, as we have seen, this is computed for the reference cell [0, 1] x [0, 1], not
        # the cell we are looping over, which is [x_{i}, x_{j}] x [y_{r}, y_{s}] and has 
        # lengths delta_x_{k} and delta_y_{l}.
        # Therefore we need to multiply each derivative by scalling coefficients to
        # correctly compute the derivative in the cell. Then, as we saw for the mass matrix,
        # we need to multiply by delta_x_{i} to correctly compute the integral. This gives
        # an overall 1/delta_x_{i} term we need to multiply, as see below.
        N_data[cell_idx, :, :] = N_local[0] * delta_y[cell_idx] / delta_x[cell_idx] + N_local[1] * delta_x[cell_idx] / delta_y[cell_idx] 

    N_global = scipy.sparse.csr_array((N_data.flatten(), (N_row_idx.flatten(), N_col_idx.flatten())), shape=(n_vertices, n_vertices))

    return N_global

def compute_forcing_term_2D(f, vertices, cells):
    r"""Computes the forcing term, right hand side, for the 2D mesh of vertices and cells.

    The forcing term F is
        F[j] = \int_{\Omega} f(x, y) B_{j}(x, y) dxdy
    
    With B_{j}(x, y) the global basis function i over the domain.
        
    Parameters
    ----------
    f : func
        The function implementing the right hand side function of the Poisson or Helmholtz equations.
    vertices : numpy.array(float), size [n+1, 1]
        The x- and y-coordinates of the vertices of the mesh. The index in the array is
        the index of the vertex, i.e., vertices[k, :] is the x- and y-coordinate of the
        k-th vertex of the mesh.
    cells : numpy.array(int), size [n, 4]
        The indices of the start and end vertex of each cell, i.e.,
        cells[k, 0] is the lower left vertex of the k-th cell
        cells[k, 1] is the lower right vertex of the k-th cell
        cells[k, 2] is the upper left vertex of the k-th cell
        cells[k, 3] is the upper right vertex of the k-th cell

    Returns
    -------
    F : numpy.array(float), size [n+1]
        The forcing term with F[j] ~= \int_{\Omega} f(x, y) B_{j}(x, y) dxdy
    """
    n_cells = cells.shape[0]
    n_vertices = vertices.shape[0]
    delta_x = (vertices[cells[:, 1], 0] - vertices[cells[:, 0], 0]).flatten()
    delta_y = (vertices[cells[:, 2], 1] - vertices[cells[:, 0], 1]).flatten()
    

    F = numpy.zeros(n_vertices)
    for cell_idx, cell in enumerate(cells):
        f_at_cell_vertices = f(vertices[cell])
        F[cell] += 0.25 * f_at_cell_vertices * delta_x[cell_idx] * delta_y[cell_idx]

    return F

def compute_solution(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y, epsilon, u, f):
    r"""
    Computes the solution to the Bubnov-Galerkin approximation for the Convection-Diffusion equation
        
        \nabla * (\epsilon \nabla\phi) - \nabla * (u \phi) = f
        \phi(\Omega) = 0


    over the interval [x_min, x_max] x [y_min, y_max] with n=n_cells uniformly distributed cells

    Parameters
    ------------
    x_min: float
        the lower bound of the interval in the x direction
    x_max: float
        the upper bound of the interval in the x direction
    y_min: float
        the lower bound of the interval in the y direction
    y_max: float
        the upper bound of the interval in the y direction
    n_cells_x: int
        the amount of cells to use in the mesh in the x direction
    n_cells_y : int
        the amount of cells to use in the mesh in the y direction
    epsilon: float
        the diffusion coefficient
    u: np.array(float), size [2]
        the velocity of the flow
    f: func (R^n_cells -> R^{n_cells})
        the function implementing the right hand side of the convection diffusion equation

    Returns
    ----
    vertices: numpy.array(float), size [(n_cells_x + 1)*(n_cells_x + 1), 2]
        The x- and y-coordinates of the vertices of the mesh. The index in the array is
        the index of the vertex, i.e., vertices[k] is the x- and y-coordinate of the
        k-th vertex of the mesh, with k given as above V_{k = i + (n_cells_x + 1)*j} = V_{i, j}.

    phi_h : numpy.array(float), size [(n_cells_x + 1)*(n_cells_x + 1)]
        The numerical solution to the convection-diffusion equation

    """

    # Impose essential (Dirichlet) boundary conditions
    phi_left = 0.0
    phi_right = 0.0
    phi_bottom = 0.0
    phi_top = 0.0
    
    # Generate the mesh
    vertices, cells = generate_mesh_2D(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y)

    # Compute global convection matrix
    M_global = compute_global_convection_matrix_2D(vertices, cells, u)
    M_global.toarray()
    
    # Compute global diffusion matrix
    N_global = compute_global_diffusion_matrix_2D(vertices, cells)
    N_global.toarray()
    

    # To be frank, I don't actually know why we construct the matrix via M_global - epsilon*N_global
    # If I derived everything correctly it should be M_global + epsilon*N_global
    # But that results in incorrect solutions. So we just add this and roll
    A = M_global - epsilon*N_global

    # Compute the right hand side
    F = compute_forcing_term_2D(f, vertices,  cells)

    # Include the boundary conditions
    # Left boundary
    i_idx = numpy.zeros(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    left_basis_indices = i_idx + (n_cells_x + 1)*j_idx
    A[left_basis_indices, :] = 0.0
    for basis_idx in left_basis_indices:
        A[basis_idx, basis_idx] = 1.0
        F[basis_idx] = phi_left  # note that phi_left is contant, if not, this needs to be changed
    
    # Right boundary
    i_idx = n_cells_x*numpy.ones(n_cells_y + 1, dtype=numpy.int64)
    j_idx = numpy.arange(0, n_cells_y + 1)
    left_basis_indices = i_idx + (n_cells_x + 1)*j_idx
    A[left_basis_indices, :] = 0.0
    for basis_idx in left_basis_indices:
        A[basis_idx, basis_idx] = 1.0
        F[basis_idx] = phi_right  # note that phi_right is contant, if not, this needs to be changed
    
    # Bottom boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = numpy.zeros(n_cells_x + 1, dtype=numpy.int64)
    left_basis_indices = i_idx + (n_cells_x + 1)*j_idx
    A[left_basis_indices, :] = 0.0
    for basis_idx in left_basis_indices:
        A[basis_idx, basis_idx] = 1.0
        F[basis_idx] = phi_bottom  # note that phi_bottom is contant, if not, this needs to be changed
    
    # Top boundary
    i_idx = numpy.arange(0, n_cells_x + 1)
    j_idx = n_cells_y*numpy.ones(n_cells_x + 1, dtype=numpy.int64)
    left_basis_indices = i_idx + (n_cells_x + 1)*j_idx
    A[left_basis_indices, :] = 0.0
    for basis_idx in left_basis_indices:
        A[basis_idx, basis_idx] = 1.0
        F[basis_idx] = phi_top  # note that phi_top is contant, if not, this needs to be changed

    # Solve the system
    phi_h = scipy.sparse.linalg.spsolve(A, F)

    return vertices, phi_h  


if __name__ == "__main__":

    # Parameters to get the same plot as in the jupyter notebook
    # x_min = 1.0
    # x_max = 2.0
    # y_min = 2.5
    # y_max = 3.0
    # N = 40

    # n_cells_x = N
    # n_cells_y = N

    # epsilon = 1
    # u = numpy.array([0.0, 0.0])
    # f = lambda point: -8.0*numpy.pi*numpy.pi*numpy.sin(2.0*numpy.pi*point[:, 0])*numpy.sin(2.0*numpy.pi*point[:, 1])  # right hand side of Poisson equation

    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0
    N = 100

    n_cells_x = N
    n_cells_y = N

    if N % 2 == 1:
        print("N is odd, this means that the even and odd solutions should be interchanged.")

    epsilon = 0.01
    u = numpy.array([0.0, 1.0])
    f = lambda points : numpy.array(list(map(lambda point : 1 if point[0] < 0.5 else -1, points)))

    vertices_even, phi_h_even = compute_solution(x_min, x_max, y_min, y_max, n_cells_x, n_cells_y, epsilon, u, f)
    vertices_odd, phi_h_odd = compute_solution(x_min, x_max, y_min, y_max, n_cells_x+1, n_cells_y+1, epsilon, u, f)

    print(len(phi_h_even))
    print(len(phi_h_odd))
    # Plot the two even-odd solutions side by side
    # plt.figure()
    # plt.pcolormesh(vertices[:, 0].reshape(n_cells_y+1, n_cells_x+1), vertices[:, 1].reshape(n_cells_y+1, n_cells_x+1), phi_h.reshape(n_cells_y+1, n_cells_x+1))  # plot the error
    # plt.colorbar()
    # plt.title("phi_h")
    # plt.show()


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].set_title('Even solution')
    im0 = ax[0].pcolormesh( vertices_even[:, 0].reshape(n_cells_y+1, n_cells_x+1), 
                            vertices_even[:, 1].reshape(n_cells_y+1, n_cells_x+1), 
                            phi_h_even.reshape(n_cells_y+1, n_cells_x+1))

    # ax[0].axvline(x=0.5, color='r', linestyle='--', linewidth=1.5)
    fig.colorbar(im0, ax=ax[0])
    
    ax[1].set_title('Odd Solution')
    im1 = ax[1].pcolormesh( vertices_odd[:, 0].reshape(n_cells_y+2, n_cells_x+2), 
                            vertices_odd[:, 1].reshape(n_cells_y+2, n_cells_x+2), 
                            phi_h_odd.reshape(n_cells_y+2, n_cells_x+2))
    fig.colorbar(im1, ax=ax[1])

    # ax[1].axvline(x=0.5, color='r', linestyle='--', linewidth=1.5)
    plt.show()
