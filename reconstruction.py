"""
    Repurposed some older code from a previous publication to generate a reconstruction tensor

    I'm assuming the angular resulution is a half degree and the energy resolution is within a factor of 5 
"""

import numpy as np
from math import sin, cos, sinh, acos, asin
from scipy.special import i0
from math import exp, pi, sqrt, log

from pone_newphs.utils import KappaGrabba, bhist

kappaCalc = KappaGrabba()

rtwolog = sqrt(2*log(2))
rtwo = 1./sqrt(2*pi)


angular_error = 0.5*pi/180
def get_odds_angle( reco, true, energy_depo ):
    kappa = kappaCalc.eval(angular_error)
    reco_angle = acos(reco)
    true_angle = acos(true)

    #return (rtwo/error)*exp(-0.5*pow((dreco-dtrue)/error,2))

    #this is the bessel part of the kent_pdf
    value = i0(kappa*sin(true_angle)*sin(reco_angle))

    # Based on Alex's Kent Distribution work 
    return kappa/(2*sinh(kappa)) * exp(kappa*cos(true_angle)*cos(reco_angle))*value*sin(reco_angle)
    
from math import log2
def get_odds_energy(deposited, reconstructed):
    """
    Takes an energy deposited and energy reconstructed.
    Loads the datafile and reads off the uncertainty from the second column. 
    The data is in %E_depo, so we scale this to be a ratio and then by that deposted energy
    """
    if not isinstance(deposited, (float,int)):
        raise Exception()
    if not isinstance(reconstructed, (float,int)):
        raise Exception()
    
    s2 = np.log(5)
    mu = np.log(np.sqrt(deposited)/2) 
    # now, we assume that the uncertainty follows a log normal distribution, and calculate the PDF here

    #prob = rtwo*(1./sigma)*exp(-0.5*((reconstructed - deposited)/sigma)**2)
    prob = rtwo*(1./sqrt(s2))*exp(-0.5*(log2(deposited) - log2(reconstructed)/5)**2)

    return(prob)

class DataReco:
    """
    This is the object that actually facilitates the energy reconstruction smearing 
    
    You pass it the edges of bins you have for energy/angle deposited/reconstructed, and then it builds up the normalized probabilities
    We keep this as a single object since building it is expensive, but accessing the probabilities is cheap! 
    """
    def __init__(self, reco_energy_edges, reco_czenith_edges, depo_energy_edges, true_czenith_edges):
        """
        Expects the energies in eV, but works in GeV
        """


        self._ereco = bhist([np.array(reco_energy_edges)*(1e-9)])
        self._edepo = bhist([np.array(depo_energy_edges)*(1e-9)])
        self._zreco = bhist([reco_czenith_edges]) # these two are in cos(Zenith)
        self._ztrue = bhist([true_czenith_edges])

        # these are now filled with the values of the probability DENSITY functions for each angle/energy combo 
        # TODO right now there is no assumed covariance ... this should be improved 
        self._energy_odds_array = np.array([[ get_odds_energy(deposited, reconstructed) for reconstructed in self.reco_energy_centers] for deposited in self.depo_energy_centers])
#        self._angle_odds_array = np.array([[ get_odds_angle(true, reconstructed) for reconstructed in self.reco_czenith_centers] for true in self.true_czenith_centers]) 
        self._angle_odds_array = np.array([[[ get_odds_angle(true, reconstructed, deposited) for reconstructed in self.reco_czenith_centers] for deposited in self.depo_energy_centers]for true in self.true_czenith_centers])

        # Avoid equating two floats. Look for a sufficiently small difference! 
        max_diff = 1e-12

        # normalize these things! 
        # so for each energy deposited... the sum of (PDF*width evaluated at each reco bin) should add up to 1. 
        for depo in range(len(self._energy_odds_array)):
            self._energy_odds_array[depo] *= 1./sum(self._energy_odds_array[depo]*self.reco_energy_widths)
            assert(abs(1-sum(self._energy_odds_array[depo]*self.reco_energy_widths)) <= max_diff)
    
        for true in range(len(self._angle_odds_array)):
            # for each of the possible true values
            for deposited in range(len(self._angle_odds_array[true])):
                # and each of the possible energies deposited
                self._angle_odds_array[true][deposited] *= 1./sum(self._angle_odds_array[true][deposited]*self.reco_czenith_widths)
                assert(abs(1-sum(self._angle_odds_array[true][deposited]*self.reco_czenith_widths)) <= max_diff)

    # these two are functions used to access those probabilities. 
    # they both take bins numbers
    # These use bin numbers _specifically_ to ensure that the user recognizes they need to use the exact same bins as they used to construct these 
    # these bin numbers should be for the bin centers/widths, NOT the edges
    def get_energy_reco_odds(self, i_depo, i_reco ):
        return(self._energy_odds_array[i_depo][i_reco]*self.reco_energy_widths[i_reco])

    def get_czenith_reco_odds(self, i_true, i_reco, i_e_true):
        return(self._angle_odds_array[i_true][i_e_true][i_reco]*self.reco_czenith_widths[i_reco])
   
    # here we have afew access functions to see what we used to build this (and what their associated bin centers/widths are)
    @property
    def reco_energy_centers(self):
        return(self._ereco.centers)
    @property
    def reco_czenith_centers(self):
        return(self._zreco.centers)
    @property
    def depo_energy_centers(self):
        return(self._edepo.centers)
    @property
    def true_czenith_centers(self):
        return(self._ztrue.centers)

    @property
    def reco_energy_widths(self):
        return(self._ereco.widths)
    @property
    def reco_czenith_widths(self):
        return(self._zreco.widths)
    @property
    def depo_energy_widths(self):
        return(self._edepo.widths)
    @property
    def true_czenith_widths(self):
        return(self._ztrue.widths)
