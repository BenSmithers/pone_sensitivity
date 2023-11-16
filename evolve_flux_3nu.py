"""
    This script isn't really used. It's just a template for basing future flux-evolution stuff off of 
"""

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--input", dest="input",
                    type=str, required=True,
                    help="Input file")

args = parser.parse_args()

infile      = args.input

prompt = False

import nuSQuIDS as nsq
import h5py as h5
import numpy as np 
from scipy.interpolate import RectBivariateSpline
import os 

from math import log10

data = h5.File(infile,'r')
energy_nodes = data["energy_nodes"][:]*(1e9) # by default, these are in GeV
costh_nodes = data["costh_nodes"][:]


parsed_data = {}
for key in data:
    parsed_data[key] = data[key][:]
data.close()
print("Loaded datafile")

n_nu = 3
n_e = 350
n_z = 100
inistate = np.zeros(shape=(n_z, n_e, 2, n_nu ))
node_e =np.logspace(2, 8, n_e)*1e9
node_czth = np.linspace(-1,1, n_z)


for i_flav in range(n_nu):
    for j_nu in range(2):
        key = ""
        if not prompt:
            key = "conv_"
        else:
            key = "pr_"

        if j_nu == 1:
            key+="antinu"
        else:
            key+="nu"
        
        if i_flav==0:
            key+="e"
        elif i_flav==1:
            key+="mu"
        elif i_flav==2:
            key+="tau"
        else:
            continue

        if key not in parsed_data:
            continue

        if (not prompt) and i_flav==2:
            pass
            #continue # don't try to get conventional taus 
        _rawflux =  np.array(parsed_data[key][:])

        double_spline = RectBivariateSpline(costh_nodes, energy_nodes, _rawflux)
        flux = double_spline(node_czth, node_e)

        for czi in range(n_z):
            for ei in range(n_e):
                inistate[czi][ei][j_nu][i_flav] += flux[czi][ei]

use_earth_interactions = True
use_oscillations = True
nus_atm =  nsq.nuSQUIDSAtm(node_czth, node_e, n_nu, nsq.NeutrinoType.both, use_earth_interactions)

xs = nsq.loadDefaultCrossSections()
nus_atm.SetNeutrinoCrossSections(xs)

nus_atm.Set_MixingAngle(0,1,0.5836)
nus_atm.Set_MixingAngle(0,2,0.1495)
nus_atm.Set_MixingAngle(1,2,0.8587)

nus_atm.Set_SquareMassDifference(1,7.42e-05)
nus_atm.Set_SquareMassDifference(2,2.51e-3)


nus_atm.Set_TauRegeneration(True)

#settting some zenith angle stuff 
nus_atm.Set_rel_error(1e-18)
nus_atm.Set_abs_error(1e-18)
nus_atm.Set_ProgressBar(True)

nus_atm.Set_GSL_step(nsq.GSL_STEP_FUNCTIONS.GSL_STEP_RKF45)
nus_atm.Set_IncludeOscillations(True)
from copy import copy
nus_atm.Set_initial_state(copy(inistate), nsq.Basis.flavor)
print("Evolving state")
nus_atm.EvolveState()

n_energy = 600
n_angle = 200
min_e = log10(min(node_e))*1.01
max_e = log10(max(node_e))*0.99

new_energies = np.logspace(min_e, max_e, n_energy)
new_zeniths = np.linspace(-1,1,n_angle)

# .EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )
outflux = {}
outflux["energy_nodes"] = new_energies/(1e9)
outflux["costh_nodes"] = new_zeniths


def flat_zen(zenith):
    zen = zenith
    if zenith<-0.9875:
        zen = -0.9875
    if zenith>0.9875:
        zen = 0.9875
    return zen

for i_flav in range(n_nu):
    for j_nu in range(2):
        key = ""
        if not prompt:
            key = "conv_"
        else:
            key = "pr_"

        if j_nu == 1:
            key+="antinu"
        else:
            key+="nu"
        
        if i_flav==0:
            key+="e"
        elif i_flav==1:
            key+="mu"
        elif i_flav==2:
            key+="tau"
        else:
            continue

        this_flux = np.array([[ 
                nus_atm.EvalFlavor(i_flav, flat_zen(zenith), energy, j_nu)
                for energy in new_energies] 
                for zenith in new_zeniths]
            )

        outflux[key] = this_flux

dirname, filename = os.path.split(infile)

outfile_name = os.path.join(
                dirname,
                ".".join(filename.split(".")[:-1]) + "_evolved.hdf5"
                )

_obj = h5.File(outfile_name, "w")
for key in outflux.keys():
    _obj.create_dataset(key, data=outflux[key])
_obj.close()