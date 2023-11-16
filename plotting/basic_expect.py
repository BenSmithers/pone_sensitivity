import numpy as np

from area_weighter import AreaWeighter, e_bins, z_bins
from pone_newphs import utils
from math import log10
import h5py as h5 

from scipy.integrate import dblquad
from scipy.interpolate import RectBivariateSpline

import matplotlib.pyplot as plt



particle_list = ['numu',   'antinumu',]

aw = AreaWeighter()
phys = utils.NSIParam()

n_evt = aw.weight_to(flux=phys)

print(np.sum(n_evt))
plt.pcolormesh(z_bins, e_bins, np.log10(n_evt.T), vmin=0,cmap ='GnBu_r')
plt.yscale('log')
plt.ylabel(r"$E_{\nu}^{true}$ [GeV]",size=14)
plt.xlabel(r"$\cos\theta_{\nu}^{true}$", size=14)
cbar = plt.colorbar()
plt.show()