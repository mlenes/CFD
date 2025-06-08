import numpy as np
from scipy.sparse.linalg import splu

class OneLevelOASPreconditioner:
    def __init__(self, A, partition):
        """Constructor

        Args:
            A (SciPy sparse matrix): The system matrix as a SciPy sparse matrix.
            partition (SciPy sparse matrix): A matrix indicating whether a node
            belongs or not the overlapping subdomain. If partition[i, j] = 1,
            then node j is in the overlapping subdomain Omega_i'.
        """
        self.A = A

        self.partition = partition
        self.num_subdomains = self.partition.shape[1]

        # The global indices of the nodes in each overlapping subdomain.
        self.Omega_ovlp = [
            self.partition[:, i].nonzero()[0] for i in range(self.num_subdomains)
        ]

        # A list containing the local solvers for each overlapping subdomain.
        self.local_solvers = [
            self._get_local_subdomain_solver(i) for i in range(self.num_subdomains)
        ]

    def _get_local_subdomain_solver(self, i: int):
        # ************
        Omega_i_ovlp = self.Omega_ovlp[i]
        Ai = self.A[Omega_i_ovlp.reshape(-1,1), Omega_i_ovlp]
        Ai_lu = splu(Ai)
        return Ai_lu
        # ************
        # Implement the initialization of the local solvers. Given the index of
        # the overlapping subdomain, construct a solver corresponding to the
        # submatrix related to the overlapping subdomain self.Omega_ovlp[i].
        # Hint: You did something similar in the previous assignment. See the code
        # for the SchwarzSolver class for reference.
        # ************

    def apply(self, x):
        # ************
        y = np.zeros_like(x)
        for i in range(self.num_subdomains):
            xi = x[self.Omega_ovlp[i]]
            yi = self.local_solvers[i].solve(xi)
            y[self.Omega_ovlp[i]] += yi
        # ************
        # Implement the application of the preconditioner to a vector x, represented
        # by a NumPy array.
        # This method implements the actual solutions on the overlapping subdomains,
        # i.e., the application of the operator
        # M^{-1} = \sum_{i = 1}^N R_i^T A_i^{-1} R_i.
        return y
        # ************


class TwoLevelOASPreconditioner(OneLevelOASPreconditioner):
    def __init__(self, A, partition, Phi):
        """Constructor

        Args:
            A (SciPy sparse matrix): The system matrix as a SciPy sparse matrix.
            partition (SciPy sparse matrix): A matrix indicating whether a node
            belongs or not the overlapping subdomain. If partition[i, j] = 1,
            then node j is in the overlapping subdomain Omega_i'.
            Phi (SciPy sparse matrix): A sparse matrix representing the prolongation
            operator Phi.
        """
        super().__init__(A, partition)
        self.Phi = Phi
        self.coarse_solver = self._get_coarse_solver()

    def _get_coarse_solver(self):
        # ************
        K0 = self.Phi.T @ self.A @ self.Phi
        K0_lu = splu(K0)
        return K0_lu
        # ************
        # Implement the initialization of the coarse space solver.
        # Hint: This is similar to the method _get_local_subdomain_solver.
        # Here, the restriction/prolongation is done with the operator \Phi
        # (self.Phi) instead of indexing the global matrix.
        # ************

    def apply(self, x):
        # ************
        coarse_rhs = self.Phi.T @ x
        coarse_sol = self.coarse_solver.solve(coarse_rhs)
        y_coarse = self.Phi @ coarse_sol

        y_fine = super().apply(x)

        return y_coarse + y_fine
        # ************
        # Implement the application of the preconditioner to a vector x, represented
        # by a NumPy array.
        # Hint: Remember that the two-level method is the same as the one-level
        # preconditioner with the added coarse space solution. You have already
        # implemented the local solution in the OneLevelOASPreconditioner class.
        # ************
