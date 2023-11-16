from pone_newphs.fitter import Fitter
from pone_newphs import utils
import numpy as np
import matplotlib.pyplot as plt

import json

param = utils.NSIParam()
test = Fitter(param)

evaled = test.evaluate_exp(np.zeros(test.dim))


e_bins = test._area.e_bins
z_bins = test._area.z_bins

plt.pcolormesh(z_bins, e_bins, np.log10(evaled  ), vmin=0,cmap ='GnBu_r')
plt.yscale('log')
plt.ylabel(r"$E_{\nu}^{reco}$ [GeV]",size=14)
plt.xlabel(r"$\cos\theta_{\nu}^{reco}$", size=14)
cbar = plt.colorbar()
plt.show()