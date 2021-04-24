#==========================================================#
#           Markov Chain sampling algorithm                #
#==========================================================#
def random_coin(p):              
    """ 
    Determines whether a candidate will be passed or not
    Parameters
    ----------
    p:
        probability of acceptance
    Returns
    -------
    True or False:
        True -> accepts; False -> rejects
    
    """
    unif = random.uniform(0,1)                         
    if unif>=p:
        return False
    else:
        return True

def mcmc_sample(hops,pdf, pdfparam):                 
    """
    Samples points for a given function with markov chain monte carlo (mcmc)
    Parameters
    ----------
    pdf: function
        function to be sampled
    n_samples: int
        number of sampled points
    pdfparam: list
        parameters that the pdf takes (if normal, standard deviation, mean...)    
        
    possible improvement: make number of walkers a controllable parameter.
    """
    states = []
    n_accepted = 0
    r_rp = np.zeros(hops)
    rp_r = np.zeros(hops)
    current = random.uniform(-4,4)
    for i in range(1,1+hops):                               # hops is the number of times you "hop", i.e. nr of steps...
        states.append(current)
        movement = current + random.uniform(-hop_size,hop_size)
        if i % n_hopsperwalker == 0 and i != 1:
            current = random.uniform(-4,4)
        curr_prob = pdf(current, *pdfparam)
        move_prob = pdf(movement, *pdfparam)            # same prob from 1 to 2 than from 2 to 1, easier in cartesian. Book by jos uses spherical -> would need some scaling 
                                                        # to account for different volume in spherical shells of different radius. Probably it would be a factor r/r'
        
        acceptance = min(move_prob/curr_prob,1)         # acceptance is A_RR'
        invacceptance = min(curr_prob/move_prob,1)
        r_rp[i-1] = acceptance * curr_prob
        rp_r[i-1] = invacceptance * move_prob
        if random_coin(acceptance) and i % n_hopsperwalker != 0:
            current = movement
            n_accepted += 1
            #print("accepted", i, "n_acc", n_accepted)
    #print ("accepted/total =", n_accepted/hops)

    return states, n_accepted/hops, r_rp, rp_r                           # give the system some time to find a reasonable starting point. 



#==========================================================#
#                    Wavefunctions/PDFs:                   #
#==========================================================#

def psi_Hydrogen(x,y,z, alpha):
    """Trial wavefunction for the Hydrogen atom's ground state 
    (in this case the trial wavefunction will be the exact wavefunction)
    Units are normalized
    Parameters
    ----------
    r:
        position r in spherical coordinates
    alpha:
        parameter of variation
    Returns
    -------
    psi:
        
"""
    r = np.sqrt(x**2+y**2+z**2)
    return alpha*r*np.exp(-alpha*r)

def psi2_Harmonic(x, alpha):
    """Trial wavefunction squared for the Hydrogen atom's ground state 
    Units are normalized
    Parameters
    ----------
    r:
        position r in spherical coordinates
    alpha:
        parameter of variation
    Returns
    -------
    psi**2 for a harmonic oscillator
    """
    return np.exp(-2*alpha*x**2)

#==========================================================#
#                     Observables/E_L                      #
#==========================================================#

def E_L_Harmonic(x, alpha):
    return alpha + x**2*(1/2-2*alpha**2)


#==========================================================#
#                       Data Processing                    #
#==========================================================#


def burn_in(states, n_removed = 4000):
    """
    removes first n_removed points of each walker's steps to ensure the final data to be equilibrated
    """
    states = np.copy(np.reshape(states, (n_walkers, n_hopsperwalker)))
    states = states[:, n_removed:]
    states = np.reshape(states, (1,n_walkers*(n_hopsperwalker-n_removed)))
    return states[0]
    
def integrate(pdff, func, param, funcparam, dist_):
    """
    Integrates a given function's product with the pdf from the mcmc_sample function
    Parameters
    ----------
    pdff: function
        this is the probability density function we want to use. (in this project it is always psi**2)
    func: function
        function we wish to find the expectation value of. (in this project this is E_L, the local energy)
    param: float
        parameter of probability density function (it will always be just alpha)
    funcparam: list
        list containing all parameters of the function (E_L) including alpha. May become just a float alpha in the future since E_L only has that parameter.
    """
    distr = np.array(dist_)
    sampled = func(distr, *funcparam)
    integral = np.average(sampled)
    err = np.var(sampled)
    return integral, err




def plot_dist(dist_, n_bins, pdf, param):
    plt.hist(dist_, density=True, bins=n_bins)
    normalization = np.sqrt(np.pi * param * 2)
    x = np.linspace(-3.5,3.5, 100)
    y = pdf(x, param)/normalization
    plt.title("Histogram of sampled points")
    plt.plot(x,y, label = "expected distribution")
    print("plot ready")
    plt.legend()
    plt.ylabel("Counts")
    plt.xlabel("x")
    plt.show()