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

def mcmc_sample_3D(hops,pdf, pdfparam):                 
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
    states_x = []
    states_y = []
    states_z = []
    n_accepted = 0
    r_rp = np.zeros(hops)
    rp_r = np.zeros(hops)
    current_x = random.uniform(-4,4)                      # generates numbers around zero in a uniform way, only for the first position of the "walker"
    current_y = random.uniform(-4,4)
    current_z = random.uniform(-4,4)
    for i in range(1,1+hops):                               # hops is the number of times you "hop", i.e. nr of steps...
        states_x.append(current_x)
        states_y.append(current_y)
        states_z.append(current_z)
        movement_x = current_x + random.uniform(-hop_size,hop_size)
        movement_y = current_y + random.uniform(-hop_size,hop_size)
        movement_z = current_z + random.uniform(-hop_size,hop_size)
        if i % n_hopsperwalker == 0 and i!=1:
            current_x = random.uniform(-4,4)
            current_y = random.uniform(-4,4)
            current_z = random.uniform(-4,4)
        curr_prob = pdf(current_x, current_y, current_z, *pdfparam)
        move_prob = pdf(movement_x, movement_y, movement_z,  *pdfparam)
        #curr_y_prob = pdf(current_y, *pdfparam)
        #move_y_prob = pdf(movement_y, *pdfparam)
        #curr_z_prob = pdf(current_z, *pdfparam)
        #move_z_prob = pdf(movement_z, *pdfparam)        # same prob from 1 to 2 than from 2 to 1, easier in cartesian. Book by jos uses spherical -> would need some scaling 
                                                        # to account for different volume in spherical shells of different radius. Probably it would be a factor r/r'
        acceptance = min(move_prob/curr_prob,1)         # acceptance is A_RR'
        invacceptance = min(curr_prob/move_prob,1)
        r_rp[i-1] = acceptance * curr_prob
        rp_r[i-1] = invacceptance * move_prob
        if random_coin(acceptance) and i % n_hopsperwalker != 0:
            current_x = movement_x
            current_y = movement_y
            current_z = movement_z
            n_accepted += 1
            #print("accepted", i, "n_acc", n_accepted)
    #print ("accepted/total =", n_accepted/hops)
    return states_x, states_y, states_z, n_accepted/hops, r_rp, rp_r                            # give the system some time to find a reasonable starting point. 



# ==========================================================#
#                    Wavefunctions/PDFs:                   #
# ==========================================================#

def normal(x,mu,sigma):                                
    #numerator = 1/sigma*np.exp(-abs(x* sigma))
    numerator = np.exp((-(x-mu)**2)/(2*sigma))
    #denominator = np.sqrt(2*np.pi*sigma)               #aka normalization constant, not needed when distribution is used for sampling
    return numerator#/denominator

def function(x):                                
    numerator = x**2
    return numerator

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
        Can it be complex?
"""
    r = np.sqrt(x**2+y**2+z**2)
    return np.exp(-2*alpha*r)

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

# ==========================================================#
#                     Observables/E_L                      #
# ==========================================================#

def E_L_Harmonic(x, alpha):
    return alpha + x**2*(1/2-2*alpha**2)

def E_L_Hydrogen(x,y,z,alpha):
    """Local energy function"""
    r = np.sqrt(x**2+y**2+z**2)
    return -1/r-1/2*alpha*(alpha-2/r)


# ==========================================================#
#                       Integration                        #
# ==========================================================#

def integrate(func, funcparam, dist_):
    """
    Integrates a given function's product with the pdf from the mcmc_sample function
    Parameters
    ----------
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

def integrate_3D(func, funcparam, dist_x_, dist_y_, dist_z_):
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
    distrx = np.array(dist_x_)
    distry = np.array(dist_y_)
    distrz = np.array(dist_z_)
    sampled = func(distrx,distry,distrz, *funcparam)
    integral = np.average(sampled)
    err = np.var(sampled)
    return integral, err  


#==========================================================#
#                     Data Processing                      #
#==========================================================#


def plot_dist(x,y,z,dist_, n_bins, pdf, param):
    
    #normalization = np.sqrt(np.pi * param * 2)
    #x = np.linspace(0,3.5, 100)
    #y = np.linspace(0,3.5, 100)
    #z = np.linspace(0,3.5, 100)
    f = pdf(x, y, z, param)#/normalization
    r = np.linspace(0,10, 500)
    f2= 4*np.pi/(np.pi/(2.4))**(3/2)*r**2*pdf(r,0,0,param)
    plt.title("Histogram of sampled points")
    #plt.scatter(np.sqrt(x**2+y**2+z**2),f, c = "r", label = "expected distribution")
    plt.plot(r, f2, label = "expected distribution", c = "r")
    plt.ylabel("Counts")
    plt.xlabel("r")

    plt.hist( dist_, density=True, bins=n_bins, label = "Sampled data")
    plt.legend()

    plt.show()
    
    

def burn_in(states, n_removed = 4000):
    """
    removes first n_removed points of each walker's steps to ensure the final data to be equilibrated
    """
    states = np.copy(np.reshape(states, (n_walkers, n_hopsperwalker)))
    states = states[:, n_removed:]
    states = np.reshape(states, (1,n_walkers*(n_hopsperwalker-n_removed)))
    return states[0]

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
