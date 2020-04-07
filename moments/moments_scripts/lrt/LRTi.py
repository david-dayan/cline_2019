import moments
import numpy


#import data
fs = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_0.8.fs")
ns = fs.sample_sizes

#define the models
# note that the parameter orders here have been moved around so match nested models, so when entering optimized params check the order
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
    nu1, nu2, T, m12, m21 = params
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
def IMg0(params, ns):
    """
    Isolation-with-migration model with split into two arbitrary sizes and subsequent gene flow with exponential growth
    p_misid: proportion of misidentified ancestral states
    
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

    
def SC0(params, ns):
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

# grab the bootstraps
all_boot = [moments.Spectrum.from_file('/home/ddayan/fundulus/moments_data/boots/SFS_{0:02d}'.format(ii))
            for ii in range(100)]

########## 
# conduct the LRT
##########

# define the simple (nested) model and its optimized params, then calculate the likelihood

# note that the parameter orders here have been moved around so match nested models, so when entering optimized params check the order

func_nest = IM0
popt_nest = [1.1864,4.0177,0.5897,0.1539,5.0822]
model_nest = func_nest(popt_nest, ns)
ll_nest = moments.Inference.ll_multinom(model_nest, fs)

#define the complex model
func = IMi0
p_opt = [2.002,6.9148,9.6479,0.2023,0.0487,2.0118,0.7864,0.3245]
model = func(p_opt, ns)
ll_model = moments.Inference.ll_multinom(model, fs)

p0 = [1.1864,4.0177,0.5897,0.1539,5.0822,0,0,0]

# look up how to do this for two parameters before proceeding
# Since LRT evaluates the complex model using the best-fit parameters from the
# simple model, we need to create list of parameters for the complex model
# using the simple (no-mig) best-fit params.  Since evalution is done with more
# complex model, need to insert zero migration value at corresponding migration
# parameter index in complex model. And we need to tell the LRT adjust function
# that the 3rd parameter (counting from 0) is the nested one.



adj = moments.Godambe.LRT_adjust(func, all_boot, p0, fs, nested_indices=[5, 6, 7], multinom=True)
D_adj = adj*2*(ll_model - ll_nest)
print('Adjusted D statistic: {0:.4f}'.format(D_adj))

