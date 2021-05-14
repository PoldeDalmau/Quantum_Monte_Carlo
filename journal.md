# Weekly progress journal

## Instructions

In this journal you will document your progress of the project, making use of weekly milestones. In contrast to project 1, you will need to define yourself detailed milestones.

Every week you should 

1. define **on Wednesday** detailed milestones for the week (according to the
   high-level milestones listed in the review issue).
   Then make a short plan of how you want to 
   reach these milestones. Think about how to distribute work in the group, 
   what pieces of code functionality need to be implemented. 
2. write about your progress **before** the Tuesday in the next week with
   respect to the milestones. Substantiate your progress with links to code,
   pictures or test results. Reflect on the relation to your original plan.

Note that there is a break before the deadline of the first week review
issue. Hence the definition of milestones and the plan for week 1 should be
done on or before 15 April.

We will give feedback on your progress on Tuesday before the following lecture. Consult the 
[grading scheme](https://computationalphysics.quantumtinkerer.tudelft.nl/proj2-grading/) 
for details how the journal enters your grade.

Note that the file format of the journal is *markdown*. This is a flexible and easy method of 
converting text to HTML. 
Documentation of the syntax of markdown can be found 
[here](https://docs.gitlab.com/ee/user/markdown.html#gfm-extends-standard-markdown). 
You will find how to include [links](https://docs.gitlab.com/ee/user/markdown.html#links) and 
[images](https://docs.gitlab.com/ee/user/markdown.html#images) particularly
useful.

## Week 1
Made a function capable of sampling following a markov chain [mcmc_sample](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L23). Computed energies for the harmonic oscillator (see Fig. 1 below) and Jos Thijssen's book ([chapter 12.2.2](Figures/table with energies.JPG)). The computations are quite slow (few minutes) samples as large as the literature's are taken (~ 15_000_000, literature uses 400*30_000 = 12_000_000, about the same...). It might be possible to remove for-loops in the mcmc_sample function. Condition for detailed balance in our code is met and is calculated [here](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L50-53) and plotted in figure 2. In the figure we show a histogram of the difference $`A_{R' R}p(R') - A_{R' R}p(R')`$. As expected, this difference is zero within python's rounding error (1e-17).


![Harmonic oscillator energies plot](Figures/Harmonic oscillator energies plot.JPG)

Fig. 1: Harmonic oscillator energies.

![Detailed Balance](Figures/Detailed balance.JPG)

Fig. 2: Histogram of $`A_{R' R}p(R') - A_{R' R}p(R')`$.

(due before 21 April)


## Week 2
(due before 28 April)


Milestones:

    - Pol: modify code to have a desired number of walkers each taking a desired number of hops.
    - Matteo: Include error of the mean calculation from previous project.
    - Alberto: Change code to sample functions of more than one variable.

Once the above is done, Hydrogen atom integrals will become possible to compute. Later, we will simply extend the number of variables from three (3D hydrogen) to six (two particles with three coordinates each in He atom). Also, we will need to implement proper minimization.

In figure 3 one can clearly see that each walker is a separate and independent object. A small hop size is used, otherwise it is too hard to discern each walker's path. What still needs to be added is a way to remove the first few steps (~3000) of each walker in order to ensure that each walker is at equilibrium.

![walkers](Figures/walkers visualized.JPG)

Fig. 3: 5 walkers with hop size = 0.01.

All the milestones have been achieved. As shown above, we can have a desired amount of walkers each taking a desired amount of hops. Also, we implemented a function that computes the [error](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L254-269) and we can sample functions in 3D with the [mcmc_sample_3D](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L61-109) (specifically, an hydrogen atom). The results for the hydrogen atom are convincing. Figure 4 below shows the histogram of the sampled distribution and the expected plot for the hydrogen atom. The two overlap, demonstrating that the simulation works correctly. Also, the results for energy given by the [Integrate_3D](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L195-215) function are in accordance with the data present in the table from the book "Computational physics" by Jos Thijssen, as shown in figure 5 below.

![Hydrogen_sample](Figures/Hydrogen - Sampled vs expected.jpeg)

Fig. 4: Histogram of sampled points for an Hydrogen atom and the expected distribution

![Hydrogen_energy_plot](Figures/Hydrogen energy plot.jpeg)

Fig. 5: Plot of energies for an Hydrogen atom for different variational paramenters



## Week 3
(due before 5 May)

Milestones:

    - Pol: calculate dE/da, calculate expectation value of dE/da
    - Alberto: mcmc function in 6D, function psi^2 for helium
    - Matteo: integrate function in 6D, update value of alpha for minimization
 
We updated the mcmc and the integrate functions to obtain the [mcmc_6D](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L61-127) and the [integrate_6D](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L397-420) functions, to work with the [He wavefunction](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L228-244) and the [local energy](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L228-244) to get results for the He ground state energy. The results, shown in figure 6 below, are in accordance with the data present in the table from the book "Computational physics" by Jos Thijssen. Generally, our values for the energy for different alphas are slightly more negative than those in the literature, but are still within the error bars.


![Helium_comparison](Figures/Helium_comparisonjpeg.JPG)

Fig. 6: Plot of energies for an Helium atom for different values of the variational parameter alpha.


Finally, we tried to implement the minimization of the energy. To do so, we implemented a function to compute [$`\frac{\mathrm{d}\ln{\psi_T}}{\mathrm{d}\alpha}`$](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L246-256) and used the integrate_6D function to obtain the expectation values. Also, we updated alpha according to the given formula. The results are as expected, the parameter alpha converges.

![Helium_minimization](Figures/Helium_minimizationjpg.JPG)

Fig 7: Minimization of the energy by tuning the parameter alpha. It converges near the point alpha = 0.20942568335742295. The iteration is repeated until the derivative of the energy with respect to alpha reaches a value below 0.001. The energy corresponding to this is around -2.89 a.u. which is in very good agreement with the literature value -2.9037243771 a.u.

Finally, we calculated the error in the energy with datablocking as we did for the first project. We see that after little more than 50 steps the energy is less and less correlated. A problem we encounter is that the error is far too small and depends on the number of sampled points. The more points we take the smaller it becomes (for a few million points the error was reduced by a factor 10). We are confident the code is correct since it worked in Project 1 so we would appreciate help on this. We do not discard the possibility that the error is in fact this small.

 ![datablocking](Figures/Error_from_datablocking.JPG)

Fig. 8: Error in the energy computed by data blocking.
