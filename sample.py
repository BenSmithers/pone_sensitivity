"""
    used earlier to get the IceCube expected event rates
"""

import numpy as np

particle_list = [
            'nue',   'antinue',
            'numu',   'antinumu',
            "nutau", "antinutau" ]

class MC_Sample:
    def __init__(self):
        self._fluxless_weight = np.array([])
        self._true_e = np.array([])
        self._true_z = np.array([])
        self._reco_z = np.array([])
        self._reco_e = np.array([])
        self._pdg = np.array([])
        self._log_true_e = np.array([])
        self._logs_evaled = False
        self._subsamples = {}
        self._cutmask = np.array([])
        
        if type(self)==MC_Sample:
            raise NotImplementedError
    def set_mask(self, mask):
        if np.shape(mask)!=np.shape(self._true_e):
            raise ValueError("Wrong mask shape")
        self._cutmask = mask

    def __getitem__(self, key)->'MC_Sample':
        return self._subsamples[key]
    @property
    def fluxless_weight(self)->np.ndarray:
        return self._fluxless_weight
    @property
    def true_e(self)->np.ndarray:
        return self._true_e
    @property
    def reco_e(self)->np.ndarray:
        return self._reco_e
    @property
    def true_z(self)->np.ndarray:
        return self._true_z
    @property
    def reco_z(self)->np.ndarray:
        return self._reco_z
    @property
    def pdg(self)->np.ndarray:
        return self._pdg
    @property
    def log_true_e(self):
        if not self._logs_evaled:
            self._log_true_e = np.log10(self._true_e)
            self._logs_evaled = True

        return self._log_true_e

class SubSample(MC_Sample):
    def __init__(self, sample:MC_Sample, mask):
        MC_Sample.__init__(self)
        self._sample = sample
        self._mask = mask
    @property
    def fluxless_weight(self)->np.ndarray:
        return self._sample.fluxless_weight[self._mask]
    @property
    def true_e(self)->np.ndarray:
        return self._sample.true_e[self._mask]
    @property
    def reco_e(self)->np.ndarray:
        return self._sample.reco_e[self._mask]
    @property
    def true_z(self)->np.ndarray:
        return self._sample.true_z[self._mask]
    @property
    def reco_z(self)->np.ndarray:
        return self._sample.reco_z[self._mask]
    @property
    def pdg(self)->np.ndarray:
        return self._sample.pdg[self._mask]
    @property
    def log_true_e(self):
        return self._sample.log_true_e[self._mask]


_mcpath = "/home/bsmithers/Downloads/IC86SterileNeutrinoDataRelease/monte_carlo/NuFSGenMC_nominal.dat"
class OneYearMC(MC_Sample):
    def __init__(self, mcpath=_mcpath):
        MC_Sample.__init__(self)
        print("Loading MC file")
        data = np.transpose(np.loadtxt(mcpath))
        self._pdg = data[0]
        self._reco_e = data[1]
        self._reco_z = data[2]
        self._true_e = data[3]
        self._true_z = data[4]
        self._fluxless_weight = data[5]/(343.7*24*3600)
        self._cutmask = np.ones_like(data[1]).astype(bool)
        self._cutmask = self._reco_z>-0.7
        #self._flux = np.zeros_like(self._fluxless_weight)

        self._subsamples={
            key:SubSample(self, self.get_mask(key)) for key in particle_list
        }

    def get_mask(self, key):

        if "tau" in key:
            mask = np.abs(self._pdg)==15
        elif "mu" in key:
            mask = np.abs(self._pdg)==13
        else:
            mask = np.abs(self._pdg)==11

        if "anti" in key:
            mask = np.logical_and(mask, self._pdg<0)
        else:
            mask = np.logical_and(mask, self._pdg>0)

        return np.logical_and(mask, self._cutmask)


        