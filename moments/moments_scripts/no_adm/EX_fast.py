import sys
import os
import numpy
import random
import moments
import pylab
from datetime import datetime
import Optimize_Functions


#create a prefix based on the population names to label the output files

prefix = "EX_ig_fast"
run = "1"

#define demo model

def demo_model(params, ns):
    """
    Split into two populations, with two migration rates. Populations are fractions of ancient
    population, where population 1 is represented by nuA*(s), and population 2 is represented by nuA*(1-s).
    Population two undergoes an exponential growth event, while population one is constant. 
	
	nuA: Ancient population size
    s: Fraction of nuA that goes to pop2. (Pop 1 has size nuA*(1-s).)
    nu1: Final size of pop 1.
    nu2: Final size of pop 2.
    T: Time in the past of split (in units of 2*Na generations) 
    m12: Migration from pop 2 to pop 1 (2*Na*m12)
    m21: Migration from pop 1 to pop 2
    """
    nuA, nu1, nu2, m12, m21, T, s = params
   
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nu2 = nuA*(1-s)
    nu1_0 = nuA*s
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2]
    fs.integrate(nu_func, T, m=asym_mig, dt_fac=0.01)
    return fs
 
#===========================================================================
# Import data to create joint-site frequency spectrum
#===========================================================================

fs = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_0.8.fs")

#print some useful information about the afs or jsfs
print("\n\n============================================================================")
print("Demographic Model: {}".format(prefix))
print("Run: {}".format(run))
print("\nData for site frequency spectrum:\n")
print("Sample sizes: {}".format(fs.sample_sizes))
print("Sum of SFS: {}".format(numpy.around(fs.S(), 2)))
print("\n============================================================================\n")



#**************
#Indicate whether your frequency spectrum object is folded (True) or unfolded (False)
fs_folded = False

"""
for models with differential introgression (include the P parameter) and models that 
attempt to control for misidentified ancestral states (p_misid), we should set some tight limits on these
parameters using a priori knowledge
Param  Min    Max
P      0.001  0.1
p_misid 0.001 0.2
this is done by passing the in_upper and in_lower args to the optimize routine function and setting them
to the limits defined below, all other params can have a wide range
"""

upper=[20,20,20,20,20,20,1]
lower=[0.001, 0.001,0.001,0.001,0.001,0.001,0.001]


#**************

#set the number of optimization rounds
rounds = 1

#define each optimization round routine
reps = [1]
maxiters = [200]
folds = [1]

# optimize
Optimize_Functions.Optimize_Routine(fs, prefix, run, demo_model, rounds, 7, fs_folded=fs_folded,
                                        reps=reps, maxiters=maxiters, folds=folds, param_labels = "nuA, nu1, nu2, m12, m21, T, s", in_upper=upper, in_lower=lower)
                                        
