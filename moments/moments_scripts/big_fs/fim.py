#check for convergence using the fim

import numpy
import moments


################
# demo model
###############
def SI(params, ns):
    """
    Split into two populations, no migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    p_misid: proportion of misidentified ancestral states
    """
    nu1, nu2, T , p_misid= params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate([nu1,nu2], T, m=nomig, dt_fac=0.01)
    #reorient misidentified ancestral alleles
    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def SIg(params, ns):
    """
    Split into two populations, no migration, population size change after split.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    p_misid: proportion of misidentified ancestral states
    """
    nu1_0, nu2_0, nu1, nu2, T , p_misid= params
    
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate(nu_func, T, m=nomig, dt_fac=0.01)
    #reorient misidentified ancestral alleles
    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def IM(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes and subsequent gene flow
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1, nu2, m12, m21, T, p_misid = params
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def IMi(params, ns):
    """
    Isolation-with-migration model with split into two arbtrary sizes, differential introgression at sites under selection
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1,nu2,T,m12,m21,m12i,m21i,P,p_misid = params


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
    return (1-p_misid)*fs2 + p_misid*moments.Numerics.reverse_array(fs2)
def IMg(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes and subsequent gene flow with exponential growth
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1_0, nu1, nu2_0, nu2, m12, m21, T, p_misid = params
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
    
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def IMig(params, ns):
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
def SC(params, ns):
    """
    Split with no gene flow, followed by period of asymmetrical gene flow

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration between pop 1 and pop 2.
    m21: migration from pop2 to pop1
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scaled time between the secondary contact and present.
    p_misid: the proportion of misindentified ancestral alleles
    """

    nu1, nu2, m12, m21, T1, T2,  p_misid= params
    
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate([nu1,nu2], T1, m=nomig, dt_fac=0.01)
    fs.integrate([nu1,nu2], T2, m=asym_mig, dt_fac=0.01)
    
    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def SC_i(params, ns):
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
    nu1, nu2, m12, m21, m12i, m21i, T1, T2, P , p_misid= params
 
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
    
    return (1-p_misid)*fs2 + p_misid*moments.Numerics.reverse_array(fs2)
def SC_g(params, ns):
    """
    Split with no gene flow with exponential growth/contraction, followed by period of asymmetrical gene flow.

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
    nu1_0, nu1, nu2_0, nu2, m12, m21, T1, T2, p_misid= params
 
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T1)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T1)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
 
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate(nu_func, T1, m=nomig, dt_fac=0.01)
    fs.integrate([nu1,nu2], T2, m=asym_mig, dt_fac=0.01)
    
    return (1-p_misid)*fs + p_misid*moments.Numerics.reverse_array(fs)
def SC_ig(params, ns):
    """
    Split with no gene flow and exponential population growth, followed by period of asymmetrical gene flow, with reduced selection at P sites.

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
    nu1_0, nu1, nu2_0, nu2, m12, m21, m12i, m21i, T1, T2, P , p_misid= params
 
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T1)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T1)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
 
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate(nu_func, T1, m=nomig, dt_fac=0.01)
    fs.integrate([nu1,nu2], T2, m=asym_mig, dt_fac=0.01)
    

    #selection
    stsi = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fsi = moments.Spectrum(stsi)
    fsi = moments.Manips.split_1D_to_2D(fsi, ns[0], ns[1])
    asym_mig_i = numpy.array([[0,m12i], [m21i,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fsi.integrate(nu_func, T1, m=nomig, dt_fac=0.01)
    fsi.integrate([nu1,nu2], T2, m=asym_mig_i, dt_fac=0.01)
   
   #combine the spectra
    fs2=P*fsi+(1-P)*fs
    
    return (1-p_misid)*fs2 + p_misid*moments.Numerics.reverse_array(fs2)
def anc(params, ns):
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
    


func_ex = anc

###########
# optimized params and data
###########

p0 = [0.4513,3.4788,1.6622,0.3426,5.9844,1.0689,2.0326,0.1976,0.6]
data = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_40_40.fs")
ns = data.sample_sizes

#######
# hessian
#######

#hess = moments.Godambe.get_hess(func_ex, p0, eps = 0.01, args=[data])
#print(hess)


##########
# fim
##########
fim = moments.Godambe.FIM_uncert(func_ex, p0, data, log=False, multinom=True, eps=0.01)
print('Estimated parameter standard deviations from FIM: {0}'.format(fim))


#############
# hessian
############
 

#moments.Godambe.get_hess(SC_ig, p0, 'eps')
