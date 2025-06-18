"""
    Uses daemonflux to get the surface fluxes 
"""

from daemonflux import Flux

import sys
from scipy import interpolate
from scipy.interpolate import fitpack2

import h5py as h5
import numpy as np
import os 

# Making a 2D flux map (E vs zenith) for a particle flux
def make2Dmap(fname, particle, vegrid, vparams={}):
    
    # Point to the files
    #quickflux = Flux('./daemonsplines_IceCube_20230207.pkl', cal_file='./prd22_daemon_v3.pkl', use_calibration=True)
    quickflux = Flux(location="generic", use_calibration=True)

    icangles = list(quickflux.zenith_angles)
    
    # Sorting the angles in the dictionary
    icangles_array = np.array(icangles, dtype=float)
    mysort = icangles_array.argsort()
    icangles = np.array(icangles)[mysort][::-1]
    icangles = icangles.astype(float)

    flux_ref = np.zeros([len(vegrid), len(icangles)])

    costheta_angles = np.zeros(len(icangles))

    for index in range(len(icangles)):
        costheta = np.cos(np.deg2rad(icangles[index]))
        costheta_angles[index] = costheta
        if fname=='conv' : 
            flux_ref[:,index] = quickflux.flux(vegrid, icangles[index], particle, params=vparams)
        elif fname=='pr'   : 
            flux_ref[:,index] = (quickflux.flux(vegrid, icangles[index], 'total_'+particle, params=vparams) - quickflux.flux(vegrid, icangles[index], particle, params=vparams))
        else:
            raise ValueError(fname)
    spl = interpolate.RectBivariateSpline(np.log10(vegrid), costheta_angles, flux_ref)  

    return spl


def makefile(fname,sname,sout) :

    params_dict = {}
    if sname!='bestfit' : params_dict[sname] = 1.0 

    print(params_dict)

    ne = 350
    nc = 100

    energies = np.logspace(2,8,ne)

    base_flux = {}
    for pname in ['numu','antinumu','nue','antinue'] : 
        base_flux[pname] = make2Dmap(fname="conv", particle=pname, vegrid=energies)
    spl_flux = {}
    for pname in ['numu','antinumu','nue','antinue'] : 
        spl_flux[pname] = make2Dmap(fname=fname, particle=pname, vegrid=energies, vparams=params_dict)

    cos_zeniths = np.linspace(-1, 1, nc)

    out_dict = {
        "energy_nodes": energies,
        "costh_nodes":cos_zeniths,
    }
    emesh, zmesh= np.meshgrid(energies, cos_zeniths)
    for pname in ['numu','antinumu','nue','antinue']:
        key=fname+"_"+pname
        out_dict[key] = np.transpose(spl_flux[pname](np.log10(energies),  cos_zeniths))/np.power(emesh,3)

        # best fit gets just the central expectation
        # the others need the difference (yields gradient here)
        if sname!="bestfit":
            out_dict[key] = out_dict[key] - np.transpose(base_flux[pname](np.log10(energies), cos_zeniths))/np.power(emesh,3)

    if sname=="bestfit":
        outfile_name = os.path.join(os.path.dirname(__file__),'surface_fluxes','daemon_'+fname+'_'+sout+'.h5')
    else:
        outfile_name =  os.path.join(os.path.dirname(__file__),'surface_fluxes','ddm_'+sout+'.h5')

    if os.path.exists(outfile_name):
        os.remove(outfile_name)
        print("Overwriting {}".format(outfile_name))
    
    _obj = h5.File(outfile_name, 'w')
    for key in out_dict.keys():
        _obj.create_dataset(key, data=out_dict[key])
    _obj.close()


makefile('conv','bestfit','bestfit')
makefile('pr','bestfit','bestfit')

syst_dict = {
    #le
    'K+_31G'   : 'le_Kplus' ,
    'K-_31G'   : 'le_Kminus' ,
    'pi+_31G'  : 'le_piplus' ,
    'pi-_31G'  : 'le_piminus' ,
    #he
    'K+_158G'  : 'he_Kplus',
    'K-_158G'  : 'he_Kminus',
    'pi+_158G' : 'he_piplus',
    'pi-_158G' : 'he_piminus',
    'n_158G'   : 'he_n',
    'p_158G'   : 'he_p',
    #vhe1
    'pi+_20T'  : 'vhe1_piplus',
    'pi-_20T'  : 'vhe1_piminus',
    #vhe3
    'K+_2P'    : 'vhe3_Kplus',
    'K-_2P'    : 'vhe3_Kminus',
    'n_2P'     : 'vhe3_n',
    'p_2P'     : 'vhe3_p',
    'pi+_2P'   : 'vhe3_piplus',
    'pi-_2P'   : 'vhe3_piminus',
    #gsf
    'GSF_1' : 'GSF_1',
    'GSF_2' : 'GSF_2',
    'GSF_3' : 'GSF_3',
    'GSF_4' : 'GSF_4',
    'GSF_5' : 'GSF_5',
    'GSF_6' : 'GSF_6'}

for sname,sout in syst_dict.items() : makefile('conv',sname,sout)