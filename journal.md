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
Pol: Made a function capable of sampling following a markov chain [mcmc_sample](https://gitlab.kwant-project.org/computational_physics/projects/Project-2---QMC_pdedalmauhugue/-/blob/master/Skeleton.py#L23). Computed energies for the harmonic oscillator (see [Figures](Figures/Harmonic oscillator energies.JPG)) as done in Jos Thijssen's book ([chapter 12.2.2](Figures/table with energies.JPG)). The computations are quite slow (few minutes) samples as large as the literature's are taken (~ 15_000_000, literature uses 400*30_000 = 12_000_000, about the same...). It might be possible to remove for-loops in the mcmc_sample function.
Next steps:
-modify code to have a desired number of walkers each taking a desired number of hops.
-Check for detailed balance
    -condition for detailed balance in our code and for cartesian coordinates (detailed balance can be checked using eq. 10.15 on page 301 from the book)
-Once the above is done, Hydrogen atom integrals can be computed.
(due before 21 April)


## Week 2
(due before 28 April)


## Week 3
(due before 5 May)


