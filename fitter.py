import numpy as np
import json
import os
import h5py as h5 

from area_weighter import AreaWeighter
from pone_newphs import utils
from nuisance import NuisanceSet, Nuisance

from scipy.optimize import minimize

from pone_newphs.reconstruction import DataReco

"""
    The fitter will load in and generate the mean expectation for each passed parameter
    Then it loads in some pre-calculated gradients
    Then it will try to fit a given realization 
"""

class Fitter:
    def __init__(self, param:utils.Param, realization_file=""):

        self._param = param
        self._area = AreaWeighter()
        self._reco_obj = DataReco(
            self._area.e_bins, self._area.z_bins,
            self._area.e_bins, self._area.z_bins
        )

        n_a = len(self._area.z_bins)-1
        n_e = len(self._area.e_bins)-1
        self._reco_tensor = [[[[ self._reco_obj.get_energy_reco_odds(j,l)*self._reco_obj.get_czenith_reco_odds(k,i,l) for i in range(n_a)] for j in range(n_e)] for k in range(n_a)] for l in range(n_e)]

        self._nominal = self._area.weight_to(param)
        self._shape = self._nominal.shape
        self._nominal = self._nominal.flatten()

        if realization_file!="":
            self._fitmode = True
            _obj = open(realization_file, 'rt')
            #self._real = np.transpose(np.array(json.load(_obj)["data"]).reshape(self._shape))
            self._real = np.array(json.load(_obj)["data"])
            
            _obj.close()
        else:
            self._fitmode = False
        

        

        self._nuisances= NuisanceSet()
        # everything must be added as multiplicative 1D gradient 
        # we assume all nuisance parameters are linear 
        # perturbed  = nominal*(1 + param*gradient)
        norm = Nuisance(0.0, 0.2, np.ones_like(self._nominal), "norm")
        self._gradient_set = None
        self._is_conv = False
        self.add_gradients([[1,],], norm)

        # load daemonflux stuff

        _covariance_file = open(os.path.join(os.path.dirname(__file__), "resources","daemon_cov.json"), 'rt')
        cov = json.load(_covariance_file)
        gradients = h5.File(os.path.join(os.path.dirname(__file__), "resources","gradients.hdf5"))
        
        
        correlation = np.zeros((len(cov.keys()), len(cov.keys())))

        cache_file = os.path.join(os.path.dirname(__file__), "resources", "cached_daemon_grad.json")
        cache_found = os.path.exists(cache_file)

        if not cache_found:
            daemon_params = []
            fc = utils.FILECLASS.CONV
            _root_folder = utils.make_root(fc)
            _convfile_name = utils.get_filename(_root_folder, "daemon_conv_bestfit.hdf5", param)
            _convdata = h5.File(_convfile_name)

        for ik, key in enumerate(cov.keys()):
            
            for jk, subkey in enumerate(cov.keys()):
                correlation[ik][jk] = cov[key][subkey]
            
            if not cache_found:            
                print("Working on {}".format(key))
                as_dict = {}
                for subkey in gradients[key].keys():
                    as_dict[subkey] = np.array(gradients[key][subkey])
                    if len(np.shape(np.array(gradients[key][subkey])))!=1:
                        as_dict[subkey]*=np.array(_convdata[subkey])
                this_grad = (self._area.weight_to(as_dict).flatten()) # /self._nominal # like, 1.2 or w/e
                this_grad/=self._nominal
                daemon_params.append(Nuisance( 0, 1, this_grad, key))

        if cache_found:
            print("loading cache")
            _obj = open(cache_file,'rt')
            data = json.load(_obj)
            _obj.close()
            daemon_params = [ Nuisance(0,1, np.array(entry), list(cov.keys())[ie]) for ie, entry in enumerate(data["cache"])]
        else:
            to_cache = {"cache": [ entry.gradient.tolist() for entry in daemon_params]}
            _obj = open(cache_file, 'wt')
            json.dump(to_cache, _obj, indent=4)
            _obj.close()

        self.add_gradients(correlation, *daemon_params)

        _covariance_file.close()
        
    @property
    def e_bins(self):
        return self._area.e_bins
    @property
    def z_bins(self):
        return self._area.z_bins

    @property
    def means(self):
        return self._nuisances.means
    @property
    def dim(self):
        return len(self.means)
    @property
    def std(self):
        return self._nuisances.stds
    @property
    def names(self):
        return self._nuisances.names


    def add_gradients(self, correlation:np.ndarray, *params:Nuisance):
        self._nuisances.add_params(correlation, *params)

        for param in params:
            if self._gradient_set is None:
                self._gradient_set = [param.gradient,]
            else:
                self._gradient_set.append(param.gradient)


    def _scipy_eval(self, params:np.ndarray)->np.ndarray:
        if not self._is_conv:
            self._gradient_set = np.array(self._gradient_set)
            self._is_conv = True
        res = self._nominal*np.prod(1+(params*self._gradient_set.T).T, axis=0)

        return res
    
    def _scipy_metric(self, params):
        exp = self.evaluate_exp(params)
        exp[exp<0] = 0
        metric = np.sum(0.5*((exp - self._real)/np.sqrt(exp))**2)

        prior_penalty = self._nuisances.prior_penalty(params)
        net =   metric+prior_penalty
        print(metric, prior_penalty, net)
        return net
    
    def evaluate_exp(self, params):
        return np.einsum('ji,klij', self._scipy_eval(params).reshape(self._shape), self._reco_tensor)
    
    def minimize(self, seeds=1):
        if not self._fitmode:
            raise NotImplementedError()
        
        start = self.means
        from scipy.optimize import basinhopping

        result = basinhopping(
            func = self._scipy_metric,
            x0=start,
            #method="L-BFGS-B",
            minimizer_kwargs={"options":{
            "ftol":1e-20,
            "gtol":1e-20,}},
            niter=10,
        )

        print(result.x)




if __name__=="__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-r", "--real",type=str,
                    help="The realization file to use")
    parser.add_argument("--mutau_real",type=float,default=0.0,
                    help="mutau real")
    parser.add_argument("--mutau_imag",type=float,default=0.0,
                    help="mutau imaginary")
    
    options = parser.parse_args()
    realization_file = options.real
    mutau_real = options.mutau_real
    mutau_imag = options.mutau_imag

    from pone_newphs import utils

    par = utils.NSIParam(mutau_real = mutau_real, mutau_imag =mutau_imag)

    fit_machine = Fitter(
        par,
        realization_file
        )
    
    fit_machine.minimize(1)