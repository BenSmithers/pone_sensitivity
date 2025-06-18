"""
    Weights a flux according to pre-defined effective areas to produce expected event rates 
"""

import os 
import h5py as h5
import numpy as np 

from pone_newphs import utils 
from pone_newphs.sample import MC_Sample

from math import log10
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import dblquad

from math import pi

particle_list = [
            'nue',   'antinue',
            'numu',   'antinumu',
            "nutau", "antinutau" ]
livetime = 5*365*24*3600
n_bin_z =  16
n_bin_e = 13

def check_load(param:utils.Param, prompt=False):
    """
        If there's a file for this parameter, return it. 
        Generate it if there is no such file. 
    """
    root_folder = utils.make_root(utils.FILECLASS.PROMPT if prompt else utils.FILECLASS.CONV)
    conv_outfile_name = utils.get_filename(root_folder, "daemon_{}_bestfit.hdf5".format("pr" if prompt else "conv"), param)
    print("Checking stuff", utils.__file__)
    print(conv_outfile_name)
    if not os.path.exists(conv_outfile_name):        
        if isinstance(param, utils.NSIParam):
            from pone_newphs.evolve_flux_nsi import main as evolve_flux
            evolve_flux(os.path.join(os.path.dirname(__file__),"surface_fluxes","daemon_{}_bestfit.h5".format("pr" if prompt else "conv")),
                        param.mutau_real,
                        param.mutau_imag,
                        prompt) 
        elif isinstance(param, utils.SterileParam):
            from pone_newphs.evolve_flux_sterile import main as evolve_flux
            evolve_flux(
                os.path.join(os.path.dirname(__file__),"surface_fluxes","daemon_{}_bestfit.h5".format("pr" if prompt else "conv")),
                param.deltam,
                param.theta14,
                param.theta24, 
                param.theta34, 
                param.deltacp41, 
                param.deltacp42, 
                prompt
            )
        else: 

            raise NotImplementedError

    return conv_outfile_name

class AreaWeighter:
    """
        Provides utilities for the effective areas. 
        Includes a function `weight_to` to get the event rates for a given flux according to these effective areas 
    """
    def __init__(self):

        self.z_bins = np.linspace(-1, 0, n_bin_z+1)
        self.e_bins = np.logspace(log10(500), log10(100000), n_bin_e+1)

        upwards = np.transpose(np.loadtxt(
            os.path.join(os.path.dirname(__file__), "eff_area","pone_-1_-0.5.dat"),
            dtype=float,
            comments="#",
            delimiter=","
        ))
        horizon = np.transpose(np.loadtxt(
            os.path.join(os.path.dirname(__file__), "eff_area","pone_-0.5_0.0.dat"),
            dtype=float,
            comments="#",
            delimiter=","
        ))
        down = np.transpose(np.loadtxt(
            os.path.join(os.path.dirname(__file__), "eff_area","pone_0.0_1.0.dat"),
            dtype=float,
            comments="#",
            delimiter=","
        ))
        # we want a continuous function taking energy/zenith and returning effective area 

        fine_energy = np.linspace(2.5, 8, 1000)

        up_fine = interp1d(np.log10(upwards[0]), np.log10(upwards[1]), fill_value="extrapolate")(fine_energy)
        horiz_fine = interp1d(np.log10(horizon[0]), np.log10(horizon[1]),fill_value="extrapolate")(fine_energy)
        down_fine = interp1d(np.log10(down[0]), np.log10(down[1]),fill_value="extrapolate")(fine_energy)

        self._areas = RectBivariateSpline(
            x=[-1, 0.0, 1.0],
            y=fine_energy,
            z=[up_fine, horiz_fine, down_fine],
            kx=1, ky=1
        )

    def weight_to(self, flux):
        """
            weights to a given flux
        """
        if isinstance(flux, str):
            return self._weight_to_str(flux)
        elif isinstance(flux, dict):
            return self._weight_to_dict(flux)
        elif isinstance(flux, utils.Param):
            return self._weight_to_param(flux) 
        else:
            raise TypeError(type(flux))
    def _weight_to_param(self, flux:utils.Param):
        #root_folder = utils.make_root(utils.FILECLASS.CONV)
        conv_outfile_name = check_load(flux, False)
        prom_outfile_name = check_load(flux, True)

        _conv_flux_raw = h5.File(conv_outfile_name,'r')
        _prom_flux_raw = h5.File(prom_outfile_name,'r')

        zeniths = np.array(_conv_flux_raw["costh_nodes"][:])
        loges = np.log10(_conv_flux_raw["energy_nodes"][:])

        mu_flux = RectBivariateSpline(zeniths, loges, np.array(_conv_flux_raw["conv_numu"])+ np.array(_prom_flux_raw["pr_numu"]))
        mubar_flux = RectBivariateSpline(zeniths, loges, np.array(_conv_flux_raw["conv_antinumu"])+np.array(_prom_flux_raw["pr_antinumu"]))


        def astro(zenith, logen):
            en = 10**logen
            return (0.787e-18)*(en/100000)**-2.5

        def funct(energy, zenith, **kwargs):
            logen = np.log10(energy)
            flux = mu_flux(zenith, logen,**kwargs) + mubar_flux(zenith, logen, **kwargs) + astro(zenith,logen)
            return self(zenith, logen)*flux*(1e4)*0.5*2*pi
        

        n_evt = np.zeros((n_bin_z, n_bin_e))

        for iz in range(n_bin_z):
            for ie in range(n_bin_e):
                value = dblquad(
                    funct, self.z_bins[iz], self.z_bins[iz+1],
                    self.e_bins[ie], self.e_bins[ie+1]
                )[0]

                n_evt[iz][ie] += value*livetime
        return n_evt

    def _weight_to_str(self, flux_str):
        _conv_flux_raw = h5.File(flux_str,'r')
        return self._weight_to_dict({
            key:np.array(_conv_flux_raw[key][:]) for key in _conv_flux_raw.keys()
        })
    def _weight_to_dict(self, _conv_flux_raw):
        zeniths = np.array(_conv_flux_raw["costh_nodes"][:])
        loges = np.log10(_conv_flux_raw["energy_nodes"][:])
        conv_interps = {}

        mu_flux = RectBivariateSpline(zeniths, loges, _conv_flux_raw["conv_numu"])
        mubar_flux = RectBivariateSpline(zeniths, loges, _conv_flux_raw["conv_antinumu"])

        def funct(energy, zenith, **kwargs):
            logen = np.log10(energy)
            flux = mu_flux(zenith, logen,**kwargs) + mubar_flux(zenith, logen, **kwargs) 
            return self(zenith, logen)*flux*(1e4)

        n_evt = np.zeros((n_bin_z, n_bin_e))

        for iz in range(n_bin_z):
            for ie in range(n_bin_e):
                value = dblquad(
                    funct, self.z_bins[iz], self.z_bins[iz+1],
                    self.e_bins[ie], self.e_bins[ie+1]
                )[0]

                n_evt[iz][ie] += value*livetime

        return n_evt

    def __call__(self, zeniths, logenergies):
        return (10**self._areas(zeniths, logenergies, grid=True))

class Weighter:
    """
        Not used right now, maybe later
    """
    def __init__(self, sample:MC_Sample, param:utils.Param):
        self._sample = sample

        # get central weights ! 
        root_folder = utils.make_root(utils.FILECLASS.CONV)
        outfile_name = utils.get_filename(root_folder, "daemon_conv_bestfit.h5", param)
        _conv_flux_raw = h5.File(outfile_name,'r')
        zeniths = np.array(_conv_flux_raw["zenith_nodes"][:])
        loges = np.log10(_conv_flux_raw["energy_nodes"][:])
        conv_interps = {}

        for key in particle_list:
            conv_interps[key] = RectBivariateSpline(zeniths, loges, _conv_flux_raw["conv_"+key])


        self._central_conv_weights = []
        for key in particle_list:
            subsamp = self._sample[key]
            weights = conv_interps[key](subsamp.true_z, subsamp.log_true_e, grid=False)*subsamp.fluxless_weight
            self._central_weights.append(weights)


    def weighted_sample(self, month:int):
        """
            Weights the sample given a certain desired month

            0 - january
            1 - february
            etc
        """

        this_interp = self._interpolators[month]

        all_weights = []

        for key in particle_list:
            subsamp = self._sample[key]
            weights = this_interp[key](subsamp.true_z, subsamp.log_true_e, grid=False)*subsamp.fluxless_weight
            all_weights += weights.tolist()
            #all_weights = np.concatenate((all_weights, weights))

        return np.sum(all_weights)