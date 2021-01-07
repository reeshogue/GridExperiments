# GridExperiments

Grid Experiment for Openended UDRL algorithm, an algorithm which uses an upside down rl algorithm and an adversary which tries to maximize the loss of the actor in order to navigate the space of an empty 32x32 grid. 

![grid.png](/grid.png "The grid wraps around...")

In my testing, the algorithm could navigate an eyeballed estimate of 20 percent of the space without rewards or explicit goals. Keep in mind that this algorithm is openended and as such does not really want to necessarily navigate the space. This is basically the UDRL version of POET. If you have any improvements, do let me know!
