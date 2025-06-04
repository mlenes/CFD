import numpy as np
import ngsolve as ngs
from ngsolve.meshes import MakeQuadMesh
from scipy.sparse import coo_matrix, csc_matrix


def assemble_fem_problem(n):
    """Auxiliary function that assembles the FEM system of equations for the 2D
    homogeneous Poisson equation.

    Args:
        n (int): Number of cells in a n x n square mesh.

    Returns:
        tuple: The assembled system matrix A and the right-hand side vector b.
    """
    mesh = MakeQuadMesh(n, n)
    fes = ngs.H1(mesh, order=1, dirichlet=".*")

    u, v = fes.TnT()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx).Assemble()
    f = ngs.LinearForm(v * ngs.dx).Assemble()

    # Convert the NGSolve matrix to a SciPy sparse matrix.
    mat_rows, mat_cols, mat_vals = a.mat.COO()
    A = coo_matrix((mat_vals, (mat_rows, mat_cols))).tocsc()

    # Assign the Dirichlet boundary conditions.
    boundary_dofs = np.nonzero(~fes.FreeDofs())[0]
    A[boundary_dofs, :] *= 0  # type: ignore
    A[:, boundary_dofs] *= 0  # type: ignore
    A[boundary_dofs, boundary_dofs] = 1
    A.eliminate_zeros()

    b = f.vec.FV().NumPy()[:]
    b[boundary_dofs] = 0

    return A, b


def assemble_prolongation_operator(N, n):
    """Assembles the prolongation operator \Phi that maps quantities from the
    coarse level to the original scale in the two-level Schwarz preconditioner.

    The grid dimensions N and n must be such that the ratio n / N is a power of 2.

    Args:
        N (int): Number of cells in the N x N coarse mesh.
        n (int): Number of cells in the n x n fine mesh.

    Returns:
        scipy.sparse.csc_matrix: The prolongation operator represented by a SciPy sparse matrix.
    """
    mesh = MakeQuadMesh(N, N)

    fes = ngs.H1(mesh, order=1, dirichlet=".*", autoupdate=True)
    u, v = fes.TnT()
    a = ngs.BilinearForm(ngs.grad(u) * ngs.grad(v) * ngs.dx)

    # The prolongation operator is computed by progressively refining
    # the coarse grid.
    levels = np.log2(n // N).astype(int)
    for _ in range(levels):
        mesh.Refine()
    a.Assemble()

    # Once we have the hierarchy of grids, we assemble the global operator
    # by successively applying the operators between each level.
    Phi_op = fes.Prolongation().Operator(1)
    for i in range(1, levels):
        Phi_op = fes.Prolongation().Operator(i + 1) @ Phi_op

    # Due to the refinement, the order of the indices on the fine grid
    # is different than what we expect. Hence, it needs to be sorted back
    # to the original order.
    coords = np.zeros((fes.ndof, 2))
    for i, vertex in enumerate(mesh.vertices):
        coords[i, 0] = vertex.point[0]
        coords[i, 1] = vertex.point[1]
    ind = np.lexsort((coords[:, 1], coords[:, 0]))

    Phi = csc_matrix(Phi_op.ToDense().NumPy()[ind, :])

    return Phi


def cg(A, b, tol=1e-6, maxiter=100, x0=None, pre=None, printrates=True):
    """Implementation of the preconditioned conjugate gradient solver.

    Args:
        A (sparse matrix): The system matrix
        b (numpy.ndarray): The right-hand side vector.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-6.
        maxiter (int, optional): Maximum number of iterations. Defaults to 100.
        x0 (numpy.ndarray, optional): The initial guess for the solution.
            If not provided, it is initialized to a vector of zeros. Defaults to None.
        pre (callable, optional): A callable object, e.g. a function, that implements the
            application of a preconditioner. Defaults to None.
        printrates(bool): Enables printing the residual at each iteration.
    Returns:
        numpy.ndarray: The converged solution.
    """
    r_curr = b - A @ x0 if x0 is not None else b.copy()
    z_curr = pre(r_curr) if pre is not None else r_curr.copy()
    p_curr = z_curr.copy() if pre is not None else r_curr.copy()
    x_curr = x0.copy() if x0 is not None else np.zeros(len(b))
    r0_l2 = np.linalg.norm(r_curr)

    for i in range(maxiter):
        alpha = np.dot(r_curr, z_curr) / np.dot(A @ p_curr, p_curr)
        x_curr += alpha * p_curr
        r_next = r_curr - alpha * A @ p_curr
        z_next = pre(r_next) if pre is not None else r_next[:]
        beta = np.dot(r_next, z_next) / np.dot(r_curr, z_curr)
        p_curr = z_next + beta * p_curr
        r_curr = r_next
        z_curr = z_next

        rel_res = np.linalg.norm(r_curr) / r0_l2

        if printrates:
            print(f"CG iteration {i + 1}, residual = {rel_res}")

        if rel_res < tol:
            break

    return x_curr
