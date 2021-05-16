# Project 2: Variational Quantum Monte Carlo
Authors: 

Pol de Dalmau Huguet (5414024), Alberto Gori (5391776) and Matteo De Luca (5388783)

How to use the code:

Only the file: Main.ipynb needs to be opened.
In the second cell, the following parameters are of interest to the user:

    -n_walkers (type:int): number of walkers that will be used for the Metropolis algorithm.
    -n_hopsperwalker (type:int): number of steps taken by each walker.
    -remove (type:int): number of steps removed from each walker for ewquilibration
    -hop_size (type: float): maximum distance a single step by a walker can be in a given cartesian coordinate x,y,z. Is to be set in order to get an acceptance ratio of about 0.5.
    -maxdEda (type: float): minimization won't stop until the derivative of energy is smaller than this number. This number couldn't be made smaller than 0.001. No convergence occurred within reasonable times. 
    -g (type: float): damping factor in steepest descent method when computing the optimal alpha. To obtain a very accurate value of alpha, we set it to 0.1. However, in this case the initial alpha must already be close to the correct one. Otherwise no convergence will occur in a reasonable amount of time.
    -onlyoneiteration (type: bool): If set to True, no minimization will take place. This will only calculate the energy, variance and datablocking error. If set to False, the minimization will go on until the condition imposed by maxdEda is satisfied. 
    
Minimization is done for Helium. A plot of all calculated values of alpha is returned. Also, a plot for the error in energy is given (only for the last)
The same is done for Hydrogen a few cells below.

The README.md file serves as a reference for other users visiting your repository.
It should contain a brief description of your project, and document the steps others need to take to get your application up and running.
In addition, it should list the authors of the project.