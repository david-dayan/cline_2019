import sys
import os
import numpy
import random
import moments
import pylab
from datetime import datetime
import Optimize_Functions


#create a prefix based on the population names to label the output files

prefix = "IMi"
run = "1"

#define demo model

def demo_model(params, ns):
    """
    Isolation-with-migration model with split into two arbtrary sizes, differential introgression at sites under selection
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1,nu2,T,m12,m21,m12i,m21i,P = params


    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1,nu2], T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    #selection
    stsi = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fsi = moments.Spectrum(stsi)
    fsi = moments.Manips.split_1D_to_2D(fsi, ns[0], ns[1])
    fsi.integrate([nu1,nu2], T, dt_fac=0.01, m=numpy.array([[0, m12i], [m21i, 0]]))
    
    #combine the spectra
    fs2=P*fsi+(1-P)*fs
    #reorient misidentified ancestral alleles
    return fs2

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

upper=[20,20,20,20,20,20,20,1]
lower=[0.001, 0.001,0.001,0.001,0.001,0.001,0.001,0.001]


#**************

#set the number of optimization rounds
rounds = 1

#define each optimization round routine
reps = [1]
maxiters = [200]
folds = [1]
params = [0.8653,5.938,5.9689,0.8599,0.1159,9.9094,2.0183,0.2582]

# optimize
Optimize_Functions.Optimize_Routine(fs, prefix, run, demo_model, rounds, 8, fs_folded=fs_folded,
                                        reps=reps, maxiters=maxiters, folds=folds, in_params=params, param_labels = "nu1,nu2,T,m12,m21,m12i,m21i,P", in_upper=upper, in_lower=lower)
                                        
