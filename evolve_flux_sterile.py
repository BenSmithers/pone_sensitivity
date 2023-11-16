"""
Evolves a flux file. 
This assumes that the file is using the standard format we get out of the MCEq scripts
    key for zenith nodes
    key for energies [GeV]
    a key for each flux 

We load that in, evolve it in nuSQuIDS, and then interpolate it using nuSQuIDS interpolator 
The output is saved to an hdf5 file 
"""

import utils

from argparse import ArgumentParser
parser = ArgumentParser()

parser.add_argument("--input", dest="input",
                    type=str, required=True,
                    help="Input file")
parser.add_argument("--dm14", dest="dm14",
                    type=float, required=True,
                    help="Delta M_41^2 - eV^2")
parser.add_argument("--th14", dest="th14",
                    type=float, required=True,
                    help="Mixing angle theta 14 [radians]")
parser.add_argument("--th24", dest="th24",
                    type=float, required=True,
                    help="Mixing angle theta 24 [radians]")
parser.add_argument("--th34", dest="th34",
                    type=float, required=True,
                    help="Mixing angle theta 34 [radians]")
parser.add_argument("--delta14", dest="delta14",
                    type=float, required=True,
                    help="CP phase 14 [radians]")
parser.add_argument("--delta24", dest="delta24",
                    type=float, required=True,
                    help="CP phase 24 [radians]")
parser.add_argument("--prompt", required=False,dest="prompt",
                    default=False, action="store_true",
                    help="This is a prompt flux")
args = parser.parse_args()

infile      = args.input
dm14        = args.dm14
th14        = args.th14
th24        = args.th24
th34        = args.th34
delta14     = args.delta14
delta24     = args.delta24
prompt      = args.prompt

print("Evolving Flux with")
print(    "theta 14 {}".format(th14))
print(    "theta 24 {}".format(th24))
print(    "theta 34 {}".format(th34))
print(    "dmsq {}".format(dm14))


import nuSQuIDS as nsq
import h5py as h5
import numpy as np
import os
from math import log10
from scipy.interpolate import RectBivariateSpline

if not os.path.exists(infile):
    raise IOError("Could not find input file {}".format(infile))

print("Loading {}".format(infile))

these_params = utils.SterileParam(
    theta14=th14,
    theta24=th24,
    theta34=th34,
    deltam=dm14,
    deltacp41=delta14,
    deltacp42=delta24

)

tolerance = 1e-15
ISSPECIAL= True
if "ddm" in infile.lower():
    fc = utils.FILECLASS.DDM
    prompt = False
    tolerance = 1e-15

else:
    fc = utils.FILECLASS.PROMT if prompt else utils.FILECLASS.CONV
print("{} type".format(fc.name))


filename_root = ".".join(os.path.split(infile)[-1].split(".")[:-1]) + ".hdf5"
root_folder = utils.make_root(fc)
outfile_name = utils.get_filename(root_folder, filename_root, these_params)

data = h5.File(infile,'r')
energy_nodes = data["energy_nodes"][:]*(1e9) # by default, these are in GeV
costh_nodes = data["costh_nodes"][:]

parsed_data = {}
for key in data:
    parsed_data[key] = data[key][:]
data.close()
print("Loaded datafile")

if all([abs(entry)<1e-15 for entry in [th14, th24, th34, dm14]]):
    n_nu = 3
    tolerance = 1e-18
else:
    n_nu = 4

if ISSPECIAL:
    n_e = 350
    n_z = 100
    inistate = np.zeros(shape=(n_z, n_e, 2, n_nu ))
    node_e =np.logspace(2, 8, n_e)*1e9
    node_czth = np.linspace(-1,1, n_z)
else:
    n_e = len(energy_nodes)
    n_z = len(costh_nodes)
    inistate = np.zeros(shape=(len(costh_nodes), len(energy_nodes), 2, n_nu ))

print("    - {}".format("Prompt" if prompt else "Conventional"))

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

        if ISSPECIAL:
            double_spline = RectBivariateSpline(costh_nodes, energy_nodes, _rawflux)
            flux = double_spline(node_czth, node_e)

        else:
            flux = _rawflux


        for czi in range(n_z):
            for ei in range(n_e):
                inistate[czi][ei][j_nu][i_flav] += flux[czi][ei]


use_earth_interactions = True
use_oscillations = True
if ISSPECIAL:
    nus_atm =  nsq.nuSQUIDSAtm(node_czth, node_e, n_nu, nsq.NeutrinoType.both, use_earth_interactions)
else:
    nus_atm =  nsq.nuSQUIDSAtm(costh_nodes, energy_nodes, n_nu, nsq.NeutrinoType.both, use_earth_interactions)
xs = nsq.loadDefaultCrossSections()
nus_atm.SetNeutrinoCrossSections(xs)

nus_atm.Set_MixingAngle(0,1,0.5836)
nus_atm.Set_MixingAngle(0,2,0.1495)
nus_atm.Set_MixingAngle(1,2,0.8587)

nus_atm.Set_SquareMassDifference(1,7.42e-05)
nus_atm.Set_SquareMassDifference(2,2.51e-3)

#sterile parameters

if n_nu==4:
    nus_atm.Set_MixingAngle(0,3,th14)
    nus_atm.Set_MixingAngle(1,3,th24)
    nus_atm.Set_MixingAngle(2,3,th34)
    nus_atm.Set_SquareMassDifference(3,dm14)
    nus_atm.Set_CPPhase(2,4, delta24)
    nus_atm.Set_CPPhase(1,4, delta14)

# set CP violating terms! 

nus_atm.Set_TauRegeneration(True)

#settting some zenith angle stuff 
nus_atm.Set_rel_error(1e-18)
print("using tolerance {}".format(tolerance))

nus_atm.Set_abs_error(tolerance)
nus_atm.Set_ProgressBar(True)

nus_atm.Set_GSL_step(nsq.GSL_STEP_FUNCTIONS.GSL_STEP_RKF45)
nus_atm.Set_IncludeOscillations(True)
from copy import copy
nus_atm.Set_initial_state(copy(inistate), nsq.Basis.flavor)
print("Evolving state")
nus_atm.EvolveState()

n_energy = 600
n_angle = 200

if ISSPECIAL:
    min_e = log10(min(node_e))*1.01
    max_e = log10(max(node_e))*0.99
else:    
    min_e = log10(min(energy_nodes))*1.01
    max_e = log10(max(energy_nodes))*0.99

new_energies = np.logspace(min_e, max_e, n_energy)
new_zeniths = np.linspace(-1,1,n_angle)

# .EvalFlavor( container["flav"], container["true_coszen"][i],  scale_e[i], i_nu )
outflux = {}
outflux["energy_nodes"] = new_energies/(1e9)
outflux["costh_nodes"] = new_zeniths

def flat_zen(zenith):
    if not ISSPECIAL:
        return zenith
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

_obj = h5.File(outfile_name, "w")
for key in outflux.keys():
    _obj.create_dataset(key, data=outflux[key])
_obj.close()