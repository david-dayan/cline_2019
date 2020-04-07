import moments
import numpy


#import data
fs = moments.Spectrum.from_file("/home/ddayan/fundulus/moments_data/SFS/fs_0.8.fs")
dd =  moments.Misc.make_data_dict('/home/ddayan/fundulus/moments_data/SFS/thinned_2d.sfs')
projections = [194,117]


# bootstrap and save
all_boot=moments.Misc.bootstrap(dd,fs.pop_ids,projections, save_dir = "/home/ddayan/fundulus/moments_data/boots/")
