"""
Evolves a flux file. 
This assumes that the file is using the standard format we get out of the MCEq scripts
    key for zenith nodes
    key for energies [GeV]
    a key for each flux 

We load that in, evolve it in nuSQuIDS, and then interpolate it using nuSQuIDS interpolator 
The output is saved to an hdf5 file 
"""


def main(infile, mutau_real, mutau_imag, prompt, force=False):

    print("Evolving Flux with")
    print(    "mutau_real {}".format(mutau_real))
    print(    "mutau_imag {}".format(mutau_imag))

    import utils


    import nuSQuIDS as nsq
    import h5py as h5
    import numpy as np
    import os
    from math import log10
    from scipy.interpolate import RectBivariateSpline

    if not os.path.exists(infile):
        raise IOError("Could not find input file {}".format(infile))

    these_params = utils.NSIParam(
        mutau_real=mutau_real,
        mutau_imag=mutau_imag,
    )

    tolerance = 1e-15
    ISSPECIAL= True
    if "ddm" in infile.lower():
        fc = utils.FILECLASS.DDM
        prompt = False
        tolerance = 1e-15

    else:
        fc = utils.FILECLASS.PROMPT if prompt else utils.FILECLASS.CONV


    filename_root = ".".join(os.path.split(infile)[-1].split(".")[:-1]) + ".hdf5"
    root_folder = utils.make_root(fc)
    outfile_name = utils.get_filename(root_folder, filename_root, these_params)
    if os.path.exists(outfile_name) and not force:
        print("Done already")
        return

    data = h5.File(infile,'r')
    energy_nodes = data["energy_nodes"][:]*(1e9) # by default, these are in GeV
    costh_nodes = data["costh_nodes"][:]

    parsed_data = {}
    for key in data:
        parsed_data[key] = data[key][:]
    data.close()

    n_nu = 3

    n_e = 500
    n_z = 100
    inistate = np.zeros(shape=(n_z, n_e, 2, n_nu ))
    node_e =np.logspace(2, 6, n_e)*1e9
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

    nus_atm =  nsq.nuSQUIDSNSI(node_czth, 0.0, node_e, n_nu, nsq.NeutrinoType.both, use_earth_interactions,
                            0.5836, 0.1495, 0.8587)

    xs = nsq.loadDefaultCrossSections()
    nus_atm.SetNeutrinoCrossSections(xs)

    nus_atm.Set_MixingAngle(0,1,0.5836)
    nus_atm.Set_MixingAngle(0,2,0.1495)
    nus_atm.Set_MixingAngle(1,2,0.8587)

    nus_atm.Set_SquareMassDifference(1,7.42e-05)
    nus_atm.Set_SquareMassDifference(2,2.51e-3)

    #nsi params
    nus_atm.Set_NSI_param(
            2,1,
            mutau_real, mutau_imag
    )

    # set CP violating terms! 

    nus_atm.Set_TauRegeneration(False)

    #settting some zenith angle stuff 
    nus_atm.Set_rel_error(1e-18)
    print("using tolerance {}".format(tolerance))

    nus_atm.Set_abs_error(tolerance)
    nus_atm.Set_ProgressBar(True)

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

    _obj = h5.File(outfile_name, "w")
    for key in outflux.keys():
        _obj.create_dataset(key, data=outflux[key])
    _obj.close()

    print("Saved {}".format(outfile_name))

if __name__=="__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--input", dest="input",
                        type=str, required=True,
                        help="Input file")
    parser.add_argument("--mutau_real", dest="mutau_real",
                        type=float, required=True,
                        help="real component of mutau")
    parser.add_argument("--mutau_imag", dest="mutau_imag",
                        type=float, required=True,
                        help="mutau_imag")
    parser.add_argument("--prompt", required=False,dest="prompt",
                        default=False, action="store_true",
                        help="This is a prompt flux")
    args = parser.parse_args()

    infile      = args.input
    mutau_real        = args.mutau_real
    mutau_imag        = args.mutau_imag

    prompt      = args.prompt

    main(infile, mutau_real, mutau_imag, prompt, True)