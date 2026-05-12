import numpy as np


class AlignmentModel:
    """Online ridge alignment (Eq. (2)–(3), McDonald 2009)."""

    def __init__(self, lambda_reg=0.1):
        self.lambda_reg = lambda_reg
        self.W = None
        self.E_local = []
        self.E_victim = []

    def update(self, local_embs, victim_embs):
        """
        local_embs: (n, d1)
        victim_embs: (n, d2)
        Eq. (3): W^t = (E_t^T E_t + λI)^{-1} E_t^T \tilde{E}_t
        """
        self.E_local.append(np.asarray(local_embs, dtype=np.float64))
        self.E_victim.append(np.asarray(victim_embs, dtype=np.float64))

        E_local = np.vstack(self.E_local)
        E_victim = np.vstack(self.E_victim)

        XtX = E_local.T @ E_local
        lamd = self.lambda_reg * np.eye(XtX.shape[0], dtype=np.float64)
        self.W = np.linalg.solve(XtX + lamd, E_local.T @ E_victim)

    def project(self, local_embs):
        if self.W is None:
            return None
        return np.asarray(local_embs, dtype=np.float64) @ self.W
