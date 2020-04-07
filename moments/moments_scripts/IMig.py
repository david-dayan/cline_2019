import sys
import os
import numpy
import random
import moments
import pylab
from datetime import datetime
import Optimize_Functions


#create a prefix based on the population names to label the output files

prefix = "IMig"
run = "1"

#define demo model

def demo_model(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes, exponential growth after split
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1_0,nu2_0,nu1,nu2,T,m12,m21,m12i,m21i,P,p_misid = params
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    #selection
    stsi = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fsi = moments.Spectrum(stsi)
    fsi = moments.Manips.split_1D_to_2D(fsi, ns[0], ns[1])
    fsi.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12i], [m21i, 0]]))
    
    #combine the spectra
    fs2=P*fsi+(1-P)*fs
    #reorient misidentified ancestral alleles
    return (1-p_misid)*fs2 + p_misid*moments.Numerics.reverse_array(fs2)

#===========================================================================
# Import data to create joint-site frequency spectrum
#===========================================================================

fs = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_55_75.fs")

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

upper=[20,20,20,20,20,20,20,20,20,1,1]
lower=[0.001, 0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.0001]


#**************

#set the number of optimization rounds
rounds = 5

#define each optimization round routine
reps = [15,15,15,20,10]
maxiters = [5,5,5,10,100]
folds = [4,3,2,1,1]

# optimize
Optimize_Functions.Optimize_Routine(fs, prefix, run, demo_model, rounds, 11, fs_folded=fs_folded,
                                        reps=reps, maxiters=maxiters, folds=folds, param_labels = "nu1_0,nu2_0,nu1,nu2,T,m12,m21,m12i,m21i,P,p_misid", in_upper=upper, in_lower=lower)
                                        
