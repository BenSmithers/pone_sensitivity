"""
This script plots a flux. 
The flux can be a nusquids one, or it can be 
"""

import nuSQuIDS as nsq
import h5py as h5 
import numpy as np 
from math import log10

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.interpolate import RectBivariateSpline

def get_nsq_neut(ptype):
    if ptype>0:
        return nsq.NeutrinoCrossSections_NeutrinoType.neutrino 
    else:
        return nsq.NeutrinoCrossSections_NeutrinoType.antineutrino
def get_nsq_flav(ptype):
    if abs(ptype)==12:
        return nsq.NeutrinoCrossSections_NeutrinoFlavor.electron
    elif abs(ptype)==14:
        return nsq.NeutrinoCrossSections_NeutrinoFlavor.muon
    elif abs(ptype)==16:
        return nsq.NeutrinoCrossSections_NeutrinoFlavor.tau
    else:
        raise ValueError("Invalide particle {}".format(ptype))

def _build_nus(filepath):
    """
        Evaluate each of the particle fluxes in a grid 
    """
    atmo = nsq.nuSQUIDSAtm(filepath)
    energies = atmo.GetERange()

    energies = np.logspace(log10(min(energies)), log10(max(energies)), 450)
    zeniths= np.linspace(-1,0, 350)

    nutype = [-1, 1]
    flavors = [12,14,16]
    outdict = {}

    for nu in nutype:
        for flavor in flavors:
            flux = [[ atmo.EvalFlavor(get_nsq_flav(flavor), 
                                      zenith, 
                                      energy,
                                      get_nsq_neut(nu))  
                            for energy in energies]
                            for zenith in zeniths]
            
            flavor_name = str(get_nsq_flav(flavor))
            if flavor_name=="electron":
                flavor_name="e"
            elif flavor_name =="muon":
                flavor_name = "mu"
            key = "antinu" if nu<0 else "nu"
            key += flavor_name

            outdict[key] = np.array(flux)
    outdict["energy_nodes"] = energies/(1e9)
    outdict["costh_nodes"] = zeniths
    outdict["is_nus"]=True

    return outdict


def _load_file(filepath):
    data = h5.File(filepath, 'r')
    if "costh_nodes" in data.keys():
        skip_keys = ["costh_nodes", "energy_nodes","is_nus"]
        out_dict = {}
        out_dict["is_nus"]=False
        for key in data.keys():
            if "pr" in key:
                continue
            if key not in skip_keys:
                nukey = str(key).split("_")[1]
            else:
                nukey = key
            print(nukey)
            out_dict[nukey] = np.array(data[key][:])
        return out_dict
    else:
        return _build_nus(filepath)


def main(flux:str, baseline="", disappear=False):
    flux_dict = _load_file(flux)

    skip_keys = ["costh_nodes", "energy_nodes","is_nus"]

    fig = plt.figure(figsize=(16,10))

    if "nutau" in flux_dict.keys():

        rank = 3
    else:
        rank = 2


    outer = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.0,width_ratios=[7,1])
    inner = gridspec.GridSpecFromSubplotSpec(2, rank, subplot_spec=outer[0], wspace=0.3, hspace=0.3)
    count = 0

    if baseline!="":
        baseline_data = _load_file(baseline)

        print("Will interpolate to broadcast these together...")

        interp_data = {}
        for key in baseline_data:
            if key in skip_keys:
                continue
            terpo = RectBivariateSpline(baseline_data["costh_nodes"], baseline_data["energy_nodes"], np.log10(baseline_data[key]))
            interp_data[key] = 10**terpo(flux_dict["costh_nodes"], flux_dict["energy_nodes"], grid=True)


        for key in flux_dict.keys():
            if key in skip_keys:
                continue

            try:
                mask = interp_data[key]!=0
                mask = np.logical_not(np.isnan(interp_data[key]))
            except KeyError:
                if "tau" in str(key):
                    flux_dict[key][mask] = flux_dict[key][mask]*0.0
                    print("Making zero {} ".format(key))
                    continue

            flux_dict[key][mask] = flux_dict[key][mask] / interp_data[key][mask] 
            flux_dict[key][np.logical_not(mask)] = 0     
            print("{} - {} to {}".format(key, np.min(flux_dict[key]), np.max(flux_dict[key])))
            vmin = 0.95
            cmap = "RdBu"

        
        vmax = 1.05
        
        cbar_label="log(Ratio)"
    else:
        vmin = -25
        vmax = -1
        cmap = "magma"
        cbar_label=r"log10($\Phi$)"
        for key in flux_dict.keys():
            if key in skip_keys:
                continue
            continue
            flux_dict[key][flux_dict[key]<0] = 0.0
            flux_dict[key][flux_dict[key]!=0] = flux_dict[key][flux_dict[key]!=0]

    for key in flux_dict.keys():
        if key in skip_keys:
            continue
        
        if disappear:
            scale = 100.
        else:
            scale = 1.

        axes = plt.Subplot(fig, inner[count])
        mesh = axes.pcolormesh(flux_dict["costh_nodes"], 
                               flux_dict["energy_nodes"], 
                               scale*np.transpose(flux_dict[key]),
                               vmin= vmin*scale, 
                               vmax=vmax*scale,
                               cmap=cmap)
        axes.set_yscale('log')
        axes.set_xlabel(r"$\cos\theta$",size=10)
        axes.set_ylabel(r"Energy [GeV]", size=10)
        axes.set_title("{}".format(key), size=10)
        axes.set_xlim([-1,0])
        axes.set_ylim([1e2, 1e8])
        fig.add_subplot(axes)

        count += 1

    which = ["nue", "numu","nutau"]

    for key in which:
        continue
        vmin= 0
        vmax = 1
        print("nu over nubar!")
        if "nutau"==key and rank==2:
            continue
        else:
            flux_one = 10**flux_dict[key] / 10**flux_dict["anti"+key]
        axes = plt.Subplot(fig, inner[count])
        mesh = axes.pcolormesh(flux_dict["costh_nodes"], 
                               flux_dict["energy_nodes"], 
                               np.transpose(flux_one) ,
                               vmin=vmin,vmax=vmax, cmap="viridis")
        axes.set_yscale('log')
        axes.set_xlabel(r"$\cos\theta$",size=10)
        axes.set_ylabel(r"Energy [GeV]", size=10)
        axes.set_title("{} nu/nubar".format(key), size=10)
        axes.set_xlim([-1,0])
        axes.set_ylim([1e2, 1e5])
        fig.add_subplot(axes)

        count+=1

        
    inner2 = gridspec.GridSpecFromSubplotSpec(1,1, subplot_spec=outer[1], wspace=0.0, hspace=0.0)
    cbar_axes = plt.Subplot(fig, inner2[0])

    cbar =plt.colorbar(mesh, cax=cbar_axes)
    if disappear:
        cbar.set_label("% Survival",size=15)
    else:
        cbar.set_label(cbar_label,size=15)
    fig.add_subplot(cbar_axes)
    print("saved {}".format("./plots/flux_plot.png"))

    plt.savefig("./plots/flux_plot.png")

    

if __name__=="__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--flux", type=str, default=None,
                    help="The flux to plot")
    parser.add_argument("--baseline", type=str, default="",
                    help="A baseline for ratio plots. Flux will be divided by this")
    parser.add_argument("--disappear", required=False,action="store_true",
                    default=False,
                    help="Flag if this is a disapperance plot, just changes label. Default: False")
    
    options = parser.parse_args()
    flux = options.flux
    baseline = options.baseline
    disappear = options.disappear

    main(flux, baseline, disappear)