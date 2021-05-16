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

def mcmc_sample_6D(hops,pdf, pdfparam):                 
    """
    Samples points for a given function with markov chain monte carlo (mcmc)
    Parameters
    ----------
    pdf: function
        function to be sampled
    hops: int
        number of sampled points
    pdfparam: list
        parameters that the pdf takes (if normal, standard deviation, mean...)
    Returns
    -------
    states_x1, states_y1, states_z1, states_x2, states_y2, states_z2: list
        sampled positions in cartesian coordinates
    n_accepted/hops: float
        acceptance ratio
    r_rp, rp_r: list
        p(R)A_RR′ and p(R')AR'R. These are later needed to check detailed balance    
    """
    states_x1 = []                                        #initialize lists for first electron
    states_y1 = []
    states_z1 = []
    states_x2 = []                                       #initialize lists for second electron
    states_y2 = []
    states_z2 = []
    n_accepted = 0
    r_rp = np.zeros(hops)
    rp_r = np.zeros(hops)
    current_x1 = random.uniform(-4,4)                      # generates numbers around zero in a uniform way, only for the first position of the "walker"
    current_y1 = random.uniform(-4,4)
    current_z1 = random.uniform(-4,4)
    current_x2 = random.uniform(-4,4)
    current_y2 = random.uniform(-4,4)
    current_z2 = random.uniform(-4,4)
    for i in range(1,1+hops):                               # hops is the number of times you "hop", i.e. nr of steps...
        states_x1.append(current_x1)
        states_y1.append(current_y1)
        states_z1.append(current_z1)
        states_x2.append(current_x2)
        states_y2.append(current_y2)
        states_z2.append(current_z2)
        movement_x1 = current_x1 + random.uniform(-hop_size,hop_size)
        movement_y1 = current_y1 + random.uniform(-hop_size,hop_size)
        movement_z1 = current_z1 + random.uniform(-hop_size,hop_size)
        movement_x2 = current_x2 + random.uniform(-hop_size,hop_size)
        movement_y2 = current_y2 + random.uniform(-hop_size,hop_size)
        movement_z2 = current_z2 + random.uniform(-hop_size,hop_size)
        if i % n_hopsperwalker == 0 and i!=1:
            current_x1 = random.uniform(-4,4)
            current_y1 = random.uniform(-4,4)
            current_z1 = random.uniform(-4,4)
            current_x2 = random.uniform(-4,4)
            current_y2 = random.uniform(-4,4)
            current_z2 = random.uniform(-4,4)
        curr_prob = pdf(current_x1, current_y1, current_z1, current_x2, current_y2, current_z2, *pdfparam)
        move_prob = pdf(movement_x1, movement_y1, movement_z1,  movement_x2, movement_y2, movement_z2, *pdfparam)       # same prob from 1 to 2 than from 2 to 1, easier in cartesian. Book by jos uses spherical -> would need some scaling 
                                                        # to account for different volume in spherical shells of different radius. Probably it would be a factor r/r'
        acceptance = min(move_prob/curr_prob,1)         # acceptance is A_RR'
        invacceptance = min(curr_prob/move_prob,1)
        r_rp[i-1] = acceptance * curr_prob
        rp_r[i-1] = invacceptance * move_prob
        if random_coin(acceptance) and i % n_hopsperwalker != 0:
            current_x1 = movement_x1
            current_y1 = movement_y1
            current_z1 = movement_z1
            current_x2 = movement_x2
            current_y2 = movement_y2
            current_z2 = movement_z2
            n_accepted += 1
            #print("accepted", i, "n_acc", n_accepted)
    #print ("accepted/total =", n_accepted/hops)
    return states_x1, states_y1, states_z1, states_x2, states_y2, states_z2, n_accepted/hops, r_rp, rp_r    
    
def mcmc_sample_3D(hops,pdf, pdfparam):                 
    """
    Samples points for a given function with markov chain monte carlo (mcmc)
    Parameters
    ----------
    pdf: function
        function to be sampled
    hops: int
        number of sampled points
    pdfparam: list
        parameters that the pdf takes (if normal, standard deviation, mean...)
    Returns
    -------
    states_x, states_y, states_z: list
        sampled positions in cartesian coordinates
    n_accepted/hops: float
        acceptance ratio
    r_rp, rp_r: list
        p(R)A_RR′ and p(R')AR'R. These are later needed to check detailed balance
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
        move_prob = pdf(movement_x, movement_y, movement_z,  *pdfparam)       # same prob from 1 to 2 than from 2 to 1, easier in cartesian. Book by jos uses spherical -> would need some scaling 
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

def psi_Hydrogen(x,y,z, alpha):
    """
    Trial wavefunction for the Hydrogen atom's ground state.    
    Units are normalized
    Parameters
    ----------
    lists: x, y, z
        positions in cartesian coordinates
    float: alpha
        parameter of variation
    Returns
    -------
    list: |psi|^2
"""
    r = np.sqrt(x**2+y**2+z**2)
    return np.exp(-2*alpha*r)


def psi2_Helium(x1,y1,z1,x2,y2,z2, alpha):
    """Trial wavefunction for the Helium atom's ground state 
    Units are normalized
    Parameters
    ----------
    x1,y1,z1,x2,y2,z2:
        Position of the two electrons in cartesian coordinates
    alpha:
        parameter of variation
    Returns
    -------
    list: |psi|^2
"""
    r_1 = np.array([x1, y1, z1])           #position vectors
    r_2 = np.array([x2, y2, z2])
    m_1 = np.sqrt(np.sum((r_1)**2, axis=0))       #modules of r_1 and r_2
    m_2 = np.sqrt(np.sum((r_2)**2, axis=0))
    r_12 = np.sqrt(np.sum((r_1 - r_2)**2, axis=0))

    return (np.exp(-4*m_1))*(np.exp(-4*m_2))*(np.exp(r_12/(1+alpha*r_12)))

def dlnpsida(x1,y1,z1,x2,y2,z2,alpha_):
    """Calculates the derivative of ln(psi_T) with respect to alpha.
     Parameters
    ----------
    x1,y1,z1, x2,y2,z2: float
        positions in cartesian coordinates of electron 1 and 2 respectively.
    alpha_: float
        variational parameter
        
    Returns
    -------
        derivative of ln(psi_T) with respect to alpha: list
    """
    r_1 = np.array([x1, y1, z1])           #position vectors
    r_2 = np.array([x2, y2, z2])
    r_12 = np.sqrt(np.sum((r_1 - r_2)**2, axis=0))
    

    return -r_12**2/(2*(r_12*alpha+1)**2)

    



# ==========================================================#
#                     Observables/E_L                      #
# ==========================================================#


def E_L_Hydrogen(x,y,z,alpha):
    """Local energy function
    
    Units are normalized
    Parameters
    ----------
    lists: x, y, z
        positions in cartesian coordinates
    float: alpha
        parameter of variation
    Returns
    -------
    list: local energy for hydrogen
    """
    r = np.sqrt(x**2+y**2+z**2)
    return -1/r-1/2*alpha*(alpha-2/r)

def E_L_Hydrogentimesr(x,y,z,alpha):
    """
    Product of local energy with negative r. Needed when computing the derivative of the energy with respect to alpha.
    
    Units are normalized
    Parameters
    ----------
    lists: x, y, z
        positions in cartesian coordinates
    float: alpha
        parameter of variation
    Returns
    -------
    Product of local energy with negative r: list
    """
    r = np.sqrt(x**2+y**2+z**2)
    return 1+r/2*alpha*(alpha-2/r)

def negativer(x,y,z,alpha):
    """
    Calculates -r. 
    Needed when computing the derivative of the energy with respect to alpha.
    
    Units are normalized
    Parameters
    ----------
    lists: x, y, z
        positions in cartesian coordinates
    float: alpha
        parameter of variation
    Returns
    -------
   negative r: list"""
    return -np.sqrt(x**2+y**2+z**2)

def E_loc(x1,y1,z1,x2,y2,z2, alpha):
    """Computes the local energy for the Helium Atom.
    Units are normalized
    Parameters
    ----------
    x1,y1,z1,x2,y2,z2:
        Position of the two electrons in cartesian coordinates
    alpha:
        parameter of variation
    Returns
    -------
    Local energy of Helium: list
    """
    r_1 = np.array([x1, y1, z1])           #position vectors
    r_2 = np.array([x2, y2, z2])
    m_1 = np.sqrt(np.sum((r_1)**2, axis=0))       #modules of r_1 and r_2
    m_2 = np.sqrt(np.sum((r_2)**2, axis=0))

    r_12 = np.sqrt(np.sum((r_1 - r_2)**2, axis=0))
    
    x_ = (x1/m_1 - x2/m_2)*(x1 - x2)
    y_ = (y1/m_1 - y2/m_2)*(y1 - y2)
    z_ = (z1/m_1 - z2/m_2)*(z1 - z2)
    
    
    return -4 + (x_ + y_ + z_) * 1/(r_12*(1 + alpha*r_12)**2) - 1/(r_12*(1 + alpha*r_12)**3) - 1/(4*r_12*(1 + alpha*r_12)**4) + 1/r_12

def E_loc_dlpsida(x1,y1,z1,x2,y2,z2, alpha):
    """
    Product of local energy with the derivative of lnpsi with respect to alpha.
    Needed when computing the derivative of the energy with respect to alpha.
    
    Units are normalized
    Parameters
    ----------
    lists: x1,y1,z1,x2,y2,z2
        positions in cartesian coordinates
    float: alpha
        parameter of variation
    Returns
    -------
    Product of local energy with the derivative of lnpsi with respect to alpha
    """
    
    E = E_loc(x1,y1,z1,x2,y2,z2, alpha)
    der = dlnpsida(x1,y1,z1,x2,y2,z2,alpha)
    
    return E*der

# ==========================================================#
#                       Integration                         #
# ==========================================================#

def integrate(func, funcparam, dist_):
    """
    Integrates a given function's product with the pdf from the mcmc_sample function
    Parameters
    ----------
    func: function
        function we wish to find the expectation value of. (in this project this is mostly the local energy)
    param: float
        parameter of probability density function (it will always be just alpha)
    funcparam: list
        list containing all parameters of the function (E_L) including alpha.
    Returns:
    -------
    integral: float
        value of integral
    err: float
        variance in the energy
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
    func: function
        function we wish to find the expectation value of. (in this project this is E_L, the local energy)
    funcparam: list
        list containing all parameters of the function (E_L) including alpha.
    dist_x_, dist_y_, dist_z_: list
        sampled distributions in cartesian coordinates
     Returns
     -------
    integral: float
        value of integral
    err: float
        variance in the energy
    """
    distrx = np.array(dist_x_)
    distry = np.array(dist_y_)
    distrz = np.array(dist_z_)
    sampled = func(distrx,distry,distrz, *funcparam)
    integral = np.average(sampled)
    err = np.var(sampled)
    return integral, err

def integrate_6D(func, funcparam, dist_x1_, dist_y1_, dist_z1_, dist_x2_, dist_y2_, dist_z2_):
    """
    Integrates a given function's product with the pdf from the mcmc_sample function
    Parameters
    ----------
    func: function
        function we wish to find the expectation value of. (in this project this is E_L, the local energy)
    funcparam: list
        list containing all parameters of the function (E_L) including alpha. May become just a float alpha in the future since E_L only has that parameter.
    dist_x1_, dist_y1_, dist_z1_, dist_x2_, dist_y2_, dist_z2_: list
        sampled distributions in cartesian coordinates
    Returns
    -------
    integral: float
        value of integral
    err: float
        variance in the energy
    """
    distrx1 = np.array(dist_x1_)
    distry1 = np.array(dist_y1_)
    distrz1 = np.array(dist_z1_)
    distrx2 = np.array(dist_x2_)
    distry2 = np.array(dist_y2_)
    distrz2 = np.array(dist_z2_)
    sampled = func(distrx1,distry1,distrz1,distrx2,distry2,distrz2, *funcparam)
    integral = np.average(sampled)
    err = np.var(sampled)
    return integral, err


# ==========================================================#
#                     Data Processing                      #
# ==========================================================#

def burn_in(states, n_removed = 4000):
    """
    removes first n_removed points of each walker's steps to ensure the final data to be equilibrated.
    """
    states = np.copy(np.reshape(states, (n_walkers, n_hopsperwalker)))
    states = states[:, n_removed:]
    states = np.reshape(states, (1,n_walkers*(n_hopsperwalker-n_removed)))
    return states[0]

def error(energies, num):
    """Computes error with datablocking for a given sample of energies.
    Parameters
    ----------
    energies: list
        list of energy at each step
    num: int
        this is the largest value of b for which one wants to calculate the error.
    Returns
    -------
    S_a: list
        values of the error
    
    """
    b = 1
    N = int(n_used)
    P = energies
    S_a = np.zeros((num, 1))
    for b in range(1, num):
        Nb = int(N/b)
        p=np.zeros(Nb)
        P=P[:(Nb*b)]
        P_shape=np.reshape(P, (Nb,b))
        P = np.copy(energies)
        p = np.sum(P_shape, axis = 1)/b
        S_a[b]=np.sqrt((1/(Nb-1))*(((1/Nb)*np.sum(np.square(p)))-np.square((1/Nb)*np.sum(p))))        #S_a[b]=np.sqrt((1/(Nb-1))*((1/b*np.sum(np.square(p)))-1/b**2*(np.sum(np.sum(p))**2)))
    return S_a

