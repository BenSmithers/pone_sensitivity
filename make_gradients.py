"""
    Takes the daemonflux gradients and calculates the in-detector gradients for each parameter
    We assume that the gradients affect the flux proportionally based off of the 3-neutrino hypothesis

    SO if some parameter is a 20% effect in some bin at the null, it'll also be a 20% effect in that same bin for some 3+1 or 3+NSI hypothesis
"""

from area_weighter import AreaWeighter
from pone_newphs.evolve_flux_nsi import main
from pone_newphs import utils

import os 
from glob import glob
import json
import h5py as h5 
import numpy as np

all_sources = glob(
    os.path.join(os.path.dirname(__file__),
    "surface_fluxes",
    "ddm_*.h5")
)

for source in all_sources:
    main(source, 0, 0, False)

_cov = open(os.path.join(os.path.dirname(__file__),"resources", "daemon_cov.json"), 'rt')
cov = json.load(_cov)
_cov.close()

all_vals = list(cov.keys())
print(len(all_vals))

all_grads = {
}

param = utils.NSIParam()

center_fname = utils.get_filename(
    utils.make_root(utils.FILECLASS.CONV), 
    "daemon_conv_bestfit.hdf5", 
    param)
center = h5.File(center_fname, 'r')

for ip, pkey in enumerate(all_vals):
    init_name = "ddm_{}.hdf5".format(pkey)

    fc = utils.FILECLASS.DDM

    root_folder = utils.make_root(fc)
    filename = utils.get_filename(root_folder, init_name, param)

    data = h5.File(filename, 'r')

    all_grads[pkey] = {}
    for key in data.keys():
        if key=="energy_nodes" or key=="costh_nodes":
            all_grads[pkey][key] = np.array(data[key][:])
        else:
            all_grads[pkey][key] = np.array(data[key][:])/np.array(center[key][:]) 

        all_grads[pkey][key] = all_grads[pkey][key].tolist()
        
outname = os.path.join(os.path.dirname(__file__),"resources", "gradients.hdf5")
_obj = h5.File(outname, 'w')
for key in all_grads.keys():
    for subkey in all_grads[key].keys():
        _obj.create_dataset("{}/{}".format(key, subkey), data=all_grads[key][subkey])


