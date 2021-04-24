def error(pdff, param):
    b = 1
    N = int(n_used)
    dist, accept_ratio, r_rp, rp_r = mcmc_sample(hops=n_samples, pdf = pdff, pdfparam=param)
    R = np.array(dist[burn_in*n_walkers:])
    #R = np.array(R)
    S_a = np.zeros((N, 1))
    for b in range(1, N):
        Nb = int(N/b)
        r=np.zeros(Nb)
        R=R[:(Nb*b)]
        R_shape=np.reshape(R, (Nb,b))
        R = dist[burn_in*n_walkers:]
        r = np.sum(R_shape, axis = 1)/b
        S_a[b]=np.sqrt((1/(Nb-1))*(((1/Nb)*np.sum(np.square(r)))-np.square((1/Nb)*np.sum(r))))
    return S_a
