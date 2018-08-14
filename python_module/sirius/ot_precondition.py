def make_kinetic_precond(kpointset, eps=0.1, asPwCoeffs=False):
    """
    Preconditioner
    P = 1 / (||k|| + ε)

    Keyword Arguments:
    kpointset --
    """
    from .coefficient_array import PwCoeffs
    import numpy as np

    nk = kpointset.num_kpoints()
    nc = kpointset.ctx().num_spins()
    if nc == 1 and nk == 1 and not asPwCoeffs:
        # return as np.matrix
        kp = kpointset[0]
        gkvec = kp.gkvec()
        assert (gkvec.num_gvec() == gkvec.count())
        N = gkvec.count()
        P = np.zeros((N, N))
        np.fill_diagonal(P, [
            1 / (np.sum(np.array(gkvec.gkvec(i))**2) + eps) for i in range(N)
        ])
        return np.matrix(P)
    else:
        P = PwCoeffs(dtype=np.float64)
        for k in range(nk):
            kp = kpointset[k]
            gkvec = kp.gkvec()
            assert (gkvec.num_gvec() == gkvec.count())
            N = gkvec.count()
            Pl = np.zeros((N, N))
            np.fill_diagonal(Pl, [
                1 / (np.sum(np.array(gkvec.gkvec(i))**2) + eps)
                for i in range(N)
            ])
            for ispn in range(nc):
                P[k, ispn] = Pl
        return P
