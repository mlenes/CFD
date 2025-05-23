import numpy as np
import ngsolve as ngs
from numpy.typing import NDArray
from ngsolve.krylovspace import LinearSolver
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


class SchwarzSolver(LinearSolver):
    def __init__(
        self, *args, partition: NDArray[np.int_], boundaries: NDArray[np.int_], **kwargs
    ):
        super().__init__(*args, **kwargs)
        if kwargs["freedofs"] is None:
            raise ValueError("freedofs must be provided")
        self.freedofs = kwargs["freedofs"]

        # A matrix indicating whether a node belongs or not the overlapping subdomain.
        # If partition[i, j] = 1, then node j is in the overlapping subdomain Omega_i'.
        self.partition = partition

        # Similar to `partition`, a matrix indicating whether a node belongs or
        # not the boundary of the overlapping subdomain. If boundaries[i, j] = 1,
        # then node j is on the boundary of the overlapping subdomain Omega_i'.
        self.boundaries = boundaries

        # Convert the NGSolve matrix to a SciPy sparse matrix so we can extract
        # the local submatrices.
        self.spmat = self._initialize_scipy_matrix()

        # The global indices of the nodes in each overlapping subdomain.
        self.Omega_ovlp = [
            self.partition[0, :].nonzero()[0],
            self.partition[1, :].nonzero()[0],
        ]


        # The global indices of the nodes on the boundary of each overlapping
        # subdomain.
        self.Omega_ovlp_boundary = [
            self.boundaries[0, :].nonzero()[0],
            self.boundaries[1, :].nonzero()[0],
        ]


        # The local indices of the nodes on the boundary of each overlapping
        # subdomain. These indices can be used to access vectors defined on
        # the overlapping subdomains.
        self.Omega_ovlp_boundary_local = [
            np.intersect1d(
                self.Omega_ovlp[0], self.Omega_ovlp_boundary[0], return_indices=True
            )[1],
            np.intersect1d(
                self.Omega_ovlp[1], self.Omega_ovlp_boundary[1], return_indices=True
            )[1],
        ]


        # The global indices of the nodes in each nonoverlapping subdomain.
        self.Omega = [
            np.setdiff1d(self.Omega_ovlp[0], self.Omega_ovlp[1]),
            np.setdiff1d(self.Omega_ovlp[1], self.Omega_ovlp[0]),
        ]


        # The global indices of the nodes that are in the overlap of the two subdomains.
        self.overlap = np.intersect1d(self.Omega_ovlp[0], self.Omega_ovlp[1])

        # A solver object representing the LU decomposition of the submatrices
        # of each overlapping subdomain.
        # You can solve a linear system defined on one of the subdomains by calling:
        # x1 = self.mat1.solve(y1)
        # where x1 and y1 are vectors defined on one of the overlapping subdomains.
        self.mat1 = self._get_local_subdomain_solver(0)
        self.mat2 = self._get_local_subdomain_solver(1)

    def _initialize_scipy_matrix(self):
        """Converts the NGSolve matrix `mat` into a SciPy sparse matrix. The
        Dirichlet boundary conditions on the system matrix are also set. By
        default, NGSolve assembles the system without setting the boundary conditions
        explicitly, later splitting the dofs into free and boundary to solve it
        more efficiently. Here, for simplicity, we consider the entire system with
        the boundary dofs.

        Returns:
            scipy.sparse.csc_matrix: A SciPy CSC sparse matrix representing `mat` with
            the Dirichlet boundary conditions set to the system.
        """
        mat_rows, mat_cols, mat_vals = self.mat.COO()
        spmat = coo_matrix((mat_vals, (mat_rows, mat_cols))).tocsc()
        spmat[~self.freedofs, :] *= 0  # type: ignore
        spmat[:, ~self.freedofs] *= 0  # type: ignore
        spmat[~self.freedofs, ~self.freedofs] = 1
        spmat.eliminate_zeros()
        return spmat

    def _get_local_subdomain_solver(self, i: int):
        """Initializes a solver object representing the LU decomposition of the
        submatrix related to the overlapping subdomain \Omega_i^'.

        Args:
            i (int): The index of the overlapping subdomain. One of 0 or 1.

        Returns:
            SuperLU object: An object representing the LU decomposition of the
            submatrix with a `solve` method.
        """
        Omega_i_ovlp = self.Omega_ovlp[i]
        Omega_i_ovlp_boundary_local = self.Omega_ovlp_boundary_local[i]

        # Extract the local overlapping subdomain matrix and set the boundary
        # conditions on the boundary of the overlapping subdomain.
        Ai = self.spmat[Omega_i_ovlp[:, None], Omega_i_ovlp]
        Ai[Omega_i_ovlp_boundary_local, :] *= 0  # type: ignore
        Ai[Omega_i_ovlp_boundary_local, Omega_i_ovlp_boundary_local] = 1
        Ai.eliminate_zeros()

        Ai_lu = splu(Ai)

        return Ai_lu


class AlternatingSchwarzSolver(SchwarzSolver):
    def _SolveImpl(self, rhs: ngs.BaseVector, sol: ngs.BaseVector):
        # Right-hand side vector restricted to the first overlapping subdomain.
        f1 = rhs.FV().NumPy()[self.Omega_ovlp[0]]

        # Right-hand side vector restricted to the second overlapping subdomain.
        f2 = rhs.FV().NumPy()[self.Omega_ovlp[1]]

        # Solution at the end of the current iteration (resp. u^{n + 1} in the assignment).
        u_curr = (np.zeros(len(rhs)),)

        # Solution at the half-step u^{n + 1/2}.
        u_half = np.zeros(len(rhs))

        while True:
            # *******************

            # 1
            f1[self.Omega_ovlp_boundary_local[0]] = u_curr[0][self.Omega_ovlp_boundary[0]] # 1.1

            u_half[self.Omega_ovlp[0]] = self.mat1.solve(f1) # 1.2

            u_half[self.Omega[1]] = u_curr[0][self.Omega[1]] # 1.3

            # 2
            f2[self.Omega_ovlp_boundary_local[1]] = u_half[self.Omega_ovlp_boundary[1]] # 2.1

            u_curr[0][self.Omega_ovlp[1]] = self.mat2.solve(f2) # 2.2

            u_curr[0][self.Omega[0]] = u_half[self.Omega[0]] # 2.3
            # *******************

            # -------

            # 1. First step of the alternating Schwarz iteration: solve the
            # problem restricted to the first overlapping subdomain.

            # 1.1. Set the boundary conditions on the boundary of the first
            # overlapping subdomain.
            # f1[...] = u_curr[...]

            # 1.2. Compute the half-step solution on the first overlapping subdomain.
            # u_half[...] = ...

            # 1.3. Update the solution on the rest of the global domain, i.e.,
            # set the value of u_half for the rest of the nodes outside \Omega_1^'
            # using u_curr.
            # u_half[...] = u_curr[...]

            # -------

            # -------

            # 2. Second step of the alternating Schwarz iteration: solve the
            # problem restricted to the second overlapping subdomain using the
            # half-step solution u^{n + 1/2}.

            # 2.1. Set the boundary conditions on the boundary of the second
            # overlapping subdomain.
            # f2[...] = u_half[...]

            # 2.2. Compute the solution on the second overlapping subdomain.
            # u_curr[...] = ...

            # 2.3. Update the solution on the first subdomain using the half-step
            # computed previously.
            # u_curr[...] = u_half[...]

            # -------

            # -------

            # 3. Convergence check: if the difference between the solutions on each
            # side in the overlap is small enough, then stop.
            r = np.linalg.norm(u_half[self.overlap] - u_curr[0][self.overlap])
            if self.CheckResidual(r):
                sol.FV().NumPy()[:] = u_curr[0]
                return

            # -------

            # *******************


class ParallelSchwarzSolver(SchwarzSolver):
    def _SolveImpl(self, rhs: ngs.BaseVector, sol: ngs.BaseVector):
        # Right-hand side vector restricted to the first overlapping subdomain.
        f1 = rhs.FV().NumPy()[self.Omega_ovlp[0]]

        # Right-hand side vector restricted to the second overlapping subdomain.
        f2 = rhs.FV().NumPy()[self.Omega_ovlp[1]]

        # Solution vectors on each overlapping subdomain.
        u1_prev, u2_prev = np.zeros(len(rhs)), np.zeros(len(rhs))
        u1_curr, u2_curr = np.zeros(len(rhs)), np.zeros(len(rhs))

        while True:
            # *******************
            # 1
            f1[self.Omega_ovlp_boundary_local[0]] = u2_prev[self.Omega_ovlp_boundary[0]] # 1.1

            u1_curr[self.Omega_ovlp[0]] = self.mat1.solve(f1) # 1.2

            # 2
            f2[self.Omega_ovlp_boundary_local[1]] = u1_prev[self.Omega_ovlp_boundary[1]] # 2.1

            u2_curr[self.Omega_ovlp[1]] = self.mat2.solve(f2) # 2.2
            # *******************

            # -------

            # 1. First step of the parallel Schwarz iteration: solve the
            # problem restricted to the first overlapping subdomain.

            # 1.1. Set the boundary conditions on the boundary of the first
            # overlapping subdomain using the solution on second overlapping
            # subdomain at the previous iteration.
            # f1[...] = u2_prev[...]

            # 1.2. Compute the updated solution on the first overlapping subdomain.
            # u1_curr[...] = ...

            # -------

            # -------

            # 2. Second step of the parallel Schwarz iteration: solve the
            # problem restricted to the second overlapping subdomain.

            # 2.1. Set the boundary conditions on the boundary of the second
            # overlapping subdomain using the solution on first overlapping
            # subdomain at the previous iteration.
            # f2[...] = u1_prev[...]

            # 2.2. Compute the updated solution on the second overlapping subdomain.
            # u2_curr[...] = ...

            # -------

            # Convergence check: if the difference between the solutions u1_curr
            # and u2_curr on the overlap of the two subdomains is small enough,
            # then stop.
            r = np.linalg.norm(u1_curr[self.overlap] - u2_curr[self.overlap])
            if self.CheckResidual(r):
                sol.FV().NumPy()[self.Omega_ovlp[0]] = u1_curr[self.Omega_ovlp[0]]
                sol.FV().NumPy()[self.Omega_ovlp[1]] = u2_curr[self.Omega_ovlp[1]]
                return

            # Update the solution vectors.
            u1_prev[:] = u1_curr
            u2_prev[:] = u2_curr

            # *******************
