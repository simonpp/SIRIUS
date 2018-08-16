def make_kinetic_precond(kpointset, c0, eps=0.1, asPwCoeffs=False):
    """
    Preconditioner
    P = 1 / (||k|| + ε)

    Keyword Arguments:
    kpointset --
    """
    from .coefficient_array import PwCoeffs
    from scipy.sparse import dia_matrix
    import numpy as np

    nk = kpointset.num_kpoints()
    nc = kpointset.ctx().num_spins()
    if nc == 1 and nk == 1 and not asPwCoeffs:
        # return as np.matrix
        kp = kpointset[0]
        gkvec = kp.gkvec()
        assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()
        d = np.array([
            1 / (np.sum(np.array(gkvec.gkvec(i))**2) + eps) for i in range(N)
        ])
        return DiagonalPreconditioner(
            D=dia_matrix((d, 0), shape=(N, N)), c0=c0)
    else:
        P = PwCoeffs(dtype=np.float64, ctype=dia_matrix)
        for k in range(nk):
            kp = kpointset[k]
            gkvec = kp.gkvec()
            assert (gkvec.num_gvec() == gkvec.count())
            N = gkvec.count()
            d = np.array([
                1 / (np.sum(np.array(gkvec.gkvec(i))**2) + eps)
                for i in range(N)
            ])
            for ispn in range(nc):
                P[k, ispn] = dia_matrix((d, 0), shape=(N, N))
        return DiagonalPreconditioner(P, c0)


class Preconditioner:
    def __init__(self):
        pass


class DiagonalPreconditioner(Preconditioner):
    """
    Apply diagonal preconditioner and project resulting gradient to satisfy the constraint.
    """
    def __init__(self, D, c0):
        super().__init__()
        self.D = D
        self.c0 = c0

    def __matmul__(self, other):
        """
        """
        from .coefficient_array import CoefficientArray
        from .ot_transformations import constrain

        out = type(other)(dtype=other.dtype)
        if isinstance(other, CoefficientArray):
            for key, Dl in self.D.items():
                out[key] = Dl * other[key]
            return constrain(out, self.c0)
        else:
            return constrain(self.D * other, self.c0)

    def __neg__(self):
        """
        """
        from .coefficient_array import CoefficientArray
        if isinstance(self.D, CoefficientArray):
            out_data = type(self.D)(dtype=self.D.dtype, ctype=self.D.ctype)
            out = DiagonalPreconditioner(out_data, self.c0)
            for k, v in self.D.items():
                out.D[k] = -v
            return out
        else:
            out = DiagonalPreconditioner(self.D, self.c0)
            out.D = -self.D
            return out
