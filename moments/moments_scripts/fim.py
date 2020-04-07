#check for convergence using the fim

import numpy
import moments


################
# demo model
###############


def SI0(params, ns):
    """
    Split into two populations, no migration.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    p_misid: proportion of misidentified ancestral states
    """
    nu1, nu2, T = params

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate([nu1,nu2], T, m=nomig, dt_fac=0.01)
    #reorient misidentified ancestral alleles
    return fs
def SIg0(params, ns):
    """
    Split into two populations, no migration, population size change after split.

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    T: Time in the past of split (in units of 2*Na generations) 
    p_misid: proportion of misidentified ancestral states
    """
    nu1_0, nu2_0, nu1, nu2, T = params
    
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]

    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate(nu_func, T, m=nomig, dt_fac=0.01)
    #reorient misidentified ancestral alleles
    return fs

def IM0(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes and subsequent gene flow
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1, nu2, m12, m21, T = params
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate([nu1, nu2], T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    return fs
def IMi0(params, ns):
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
def IMg0(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes and subsequent gene flow with exponential growth

    """
    nu1_0, nu1, nu2_0, nu2, m12, m21, T= params
    nu1_func = lambda t: nu1_0 * (nu1/nu1_0)**(t/T)
    nu2_func = lambda t: nu2_0 * (nu2/nu2_0)**(t/T)
    nu_func = lambda t: [nu1_func(t), nu2_func(t)]
    
    #neutral
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    fs.integrate(nu_func, T, dt_fac=0.01, m=numpy.array([[0, m12], [m21, 0]]))

    return fs
def IMig0(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes, exponential growth after split
    p_misid: proportion of misidentified ancestral states
    
    """
    nu1_0,nu2_0,nu1,nu2,T,m12,m21,m12i,m21i,P = params
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
    return fs2
    
def SC0(params, ns):
    """
    Split with no gene flow, followed by period of asymmetrical gene flow

    nu1: Size of population 1 after split.
    nu2: Size of population 2 after split.
    m12: Migration between pop 1 and pop 2.
    m21: migration from pop2 to pop1
    T1: The scaled time between the split and the secondary contact (in units of 2*Na generations).
    T2: The scaled time between the secondary contact and present.
    """

    nu1, nu2, m12, m21, T1, T2= params
    
    sts = moments.LinearSystem_1D.steady_state_1D(ns[0] + ns[1])
    fs = moments.Spectrum(sts)
    fs = moments.Manips.split_1D_to_2D(fs, ns[0], ns[1])
    asym_mig = numpy.array([[0,m12], [m21,0]])
    nomig = numpy.array([[0,0], [0,0]])
    fs.integrate([nu1,nu2], T1, m=nomig, dt_fac=0.01)
    fs.integrate([nu1,nu2], T2, m=asym_mig, dt_fac=0.01)
    
    return fs
def SCi0(params, ns):
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
def SCig0(params, ns):
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
    nu1_0, nu1, nu2_0, nu2, m12, m21, m12i, m21i, T1, T2, P = params
 
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
    
    return fs2
def SCg0(params, ns):
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
    nu1_0, nu1, nu2_0, nu2, m12, m21, T1, T2= params
 
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
    
    return fs

    


func_ex = SIg0

###########
# optimized params and data
###########

p0 = [	0.0456,1.4568,0.5341,0.3352,0.0333]
data = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_0.8.fs")
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
