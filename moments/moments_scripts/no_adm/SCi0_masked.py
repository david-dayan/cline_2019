import sys
import os
import numpy
import random
import moments
import pylab
from datetime import datetime
import Optimize_Functions


#create a prefix based on the population names to label the output files

prefix = "SCi"
run = "1"

#define demo model

def demo_model(params, ns):
    """
    Split with no gene flow, followed by period of asymmetrical gene flow, with reduced selection at P sites.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration between pop 1 and pop 2.
    m21: migration from pop2 to pop1
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scaled time between the secondary contact and present.
    P: neutral portion
    m12i: migration at selected sites
    m21i: migration at selected sites
    """
    nu1, nu2, m12, m21, m12i, m21i, T1, T2, P= params
 
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate([nu1,nu2], T1, m=nomig, dt_fac=0.01)
    fs.integrate([nu1,nu2], T2, m=asym_mig, dt_fac=0.01)
    
    #selection
    stsi = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fsi = moments.Spectrum(stsi)
    fsi = moments.Manips.split_1D_to_2D(fsi, ns[0], ns[1])
    asym_mig_i = numpy.array([[0,m12i], [m21i,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fsi.integrate([nu1,nu2], T1, m=nomig, dt_fac=0.01)
    fsi.integrate([nu1,nu2], T2, m=asym_mig_i, dt_fac=0.01)
   
   #combine the spectra
    fs2=P*fsi+(1-P)*fs
    
    return fs2

#===========================================================================
# Import data to create joint-site frequency spectrum
#===========================================================================

fs = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/masked.fs")

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

upper=[20,20,20,20,20,20,20,20,1]
lower=[0.001, 0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001]


#**************

#set the number of optimization rounds
rounds = 5

#define each optimization round routine
reps = [15,10,10,10,1]
maxiters = [5,5,5,50,200]
folds = [4,3,2,1,1]

# optimize
Optimize_Functions.Optimize_Routine(fs, prefix, run, demo_model, rounds, 9, fs_folded=fs_folded,
                                        reps=reps, maxiters=maxiters, folds=folds,  param_labels = "nu1, nu2, m12, m21, m12i, m21i, T1, T2, P , p_misid", in_upper=upper, in_lower=lower)
                                        
