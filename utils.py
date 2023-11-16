import numpy as np
from enum import Enum
import os 
import pickle 
from math import exp,sinh, cos
import scipy.optimize as opt


class FILECLASS(Enum):
    """
    If this is a realization, a flux file, a fit, etc
    """
    CONV = 0
    PROMPT = 1
    ASTR = 2
    FIT = 3
    DDM = 4
    REAL = 5


# where it will look for fluxes
RESOUCES_FOLDER = "/home/bsmithers/software/data/pone"

# where it will save fit files 
OUT_FOLDER = "/home/bsmithers/software/data/pone"




def make_root(fc:FILECLASS, batchname="", osg=False)->str:
    """
    Builds up the root folder for file storage using the track/cascade, flux/fit/real, stuff as a way of sorting them 
    """
    if fc==FILECLASS.FIT and batchname=="":
        raise ValueError("Fits need a batch name")  
    else:
        if fc==FILECLASS.FIT:
            if not os.path.exists(os.path.join(OUT_FOLDER, fc.name, batchname)):
                try:
                    os.mkdir(os.path.join(OUT_FOLDER, fc.name, batchname))
                except:
                    print("Failed making a directory! {}".format(os.path.join(OUT_FOLDER, fc.name, batchname)))
            folder = os.path.join(OUT_FOLDER, fc.name, batchname)
            if not os.path.exists(folder):
                try:
                    os.mkdir(folder)
                except:
                    print("Failed making a directory! {}".format(folder))
        else: 
            folder = os.path.join(OUT_FOLDER,fc.name)

    if not os.path.exists(folder):
        try:
            os.mkdir(folder)
        except:
            print("Failed making a directory! {}".format(folder))

    return folder 


class Param:
    """
        Generic class for new physics parameters. Might be able to use this if I ever do an NSI or w/e analysis 
    """
    __state_params = [] # parameters needed to describe the new physics model 
    __kind = "param" # used in naming to distinguish different physics models 
    __subfolder = ""

    def __init__(self, **kwargs):
        """
            Sets each of the parameters in `state_params` as an attribute for this object 

            If the parameter was provided as a keyword argument, set the value to the given value. Otherwise make it zero 

            Then we protect those values by replacing each of those with a function that returns the value 
        """
        for item in self.state_params():
            if item in kwargs:
                setattr(self, item, kwargs[item])
            else:
                setattr(self, item, 0.0)

        for item in self.state_params():
            def temp():
                return getattr(self, item)
            
            setattr(self, item, temp())

    
    def __str__(self)->str:
        """
            Makes a string that encapsulates all the values needed to describe this new physics point 
        """
        return self.kind() +"_"+ "_".join( ["{:.6E}".format(getattr(self, item)) for item in self.state_params()] )

    @classmethod
    def state_params(cls):
        """
            Returns the list of parameters describing the point 
        """
        return cls.__state_params
    
    @classmethod
    def kind(cls):
        """
            Returns the `__kind` property
        """
        return cls.__kind
    
    @classmethod
    def has_subfolder(cls):
        return False
    
    @classmethod 
    def subfolder(cls):
        """
            Parameters can return a name. This is used so that we can sub-divide the parameter files into sub-folders so that the lfs7 file system doesn't get sad 
        """
        raise NotImplementedError("")

def get_filename(root_folder:str, name_root:str, param:Param)->str:
    """
        Used for producing filenames and putting them in subfolders (if desired)
    """

    ## trust that the user doesn't use a lot of stacked extensions

    broken = name_root.split(".")
    name = ".".join( broken[:-1] )
    ext = broken[-1]

    new_name = name + "_{}".format(param) + "." + ext

    if not param.has_subfolder():
        ret_filename = os.path.join(root_folder, new_name)
    else:
        if not os.path.exists(os.path.join(root_folder, param.subfolder())) and root_folder!="":
            try:
                os.mkdir(os.path.join(root_folder, param.subfolder()))
            except:
                print("Failed making a directory! {}".format(os.path.join(root_folder, param.subfolder())))
        ret_filename = os.path.join(root_folder, param.subfolder(), new_name)

    return ret_filename
    

class SterileParam(Param):
    """
    deltam
    theta14
    theta24
    theta34
    deltacp41
    deltacp42

    Implementation for Sterile neutrino oscillations 
    """
    __state_params = [
        "deltam",
        "theta14",
        "theta24",
        "theta34",
        "deltacp41",
        "deltacp42"
    ]

    __kind = "sterile"
    __subfolder = __state_params[0]

    @classmethod
    def state_params(cls):
        return cls.__state_params
    
    @classmethod
    def kind(cls):
        return cls.__kind
    
    @classmethod
    def has_subfolder(cls):
        return True

    def subfolder(self):
        return "{:.4E}".format(getattr(self, SterileParam.__subfolder))

class NSIParam(Param):
    """
    emu_real
    emu_imag,
    etau_real,
    etau_imag,
    mutau_real
    mutau_imag

    I prepared this and added it here in case it's useful in the future 
    """
    __state_params=[
        "emu_real",
        "emu_imag",
        "etau_real",
        "etau_imag",
        "mutau_real",
        "mutau_imag"
    ]

    __kind="nsi"
    __subfolder = __state_params[-2] #eps mutau real

    @classmethod
    def state_params(cls):
        return cls.__state_params
    
    @classmethod
    def kind(cls):
        return cls.__kind
    
    @classmethod
    def has_subfolder(cls):
        return True

    def subfolder(self):
        return "{:.4E}".format(getattr(self, NSIParam.__subfolder))




def get_color(n, colormax=3.0, cmap="viridis"):
    """
        Discretize a colormap. Great getting nice colors for a series of trends on a single plot! 
    """
    import matplotlib.pyplot as plt

    this_cmap = plt.get_cmap(cmap)
    return this_cmap(n/colormax)

def get_loc(x, domain,closest=False):
    """
    Implements a binary search to return the indices of the entries in domain that border 'x' 
    Raises exception if x is outside the range of domain 

    Assumes 'domain' is sorted!! And this _only_ works if the domain is length 2 or above 

    This is made for finding bin numbers on a list of bin edges, and is used mainly by the fastmode scaling code.
    """
    if not isinstance(domain, (tuple,list,np.ndarray)):
        raise TypeError("'domain' has unrecognized type {}, try {}".format(type(domain), list))
    if not isinstance(x, (float,int)):
        raise TypeError("'x' should be number-like, not {}".format(type(x)))
    
    if len(domain)<=1:
        raise ValueError("get_loc function only works on domains of length>1. This is length {}".format(len(domain)))

    if x<domain[0] or x>domain[-1]:
        raise ValueError("x={} and is outside the domain: ({}, {})".format(x, domain[0], domain[-1]))

    min_abs = 0
    max_abs = len(domain)-1

    lower_bin = int(abs(max_abs-min_abs)/2)
    upper_bin = lower_bin+1
    

    while not (domain[lower_bin]<x and domain[upper_bin]>=x):
        if abs(max_abs-min_abs)<=1:
            print("{} in {}".format(x, domain))
            
            raise Exception("Uh Oh")

        if x<domain[lower_bin]:
            max_abs = lower_bin
        if x>domain[upper_bin]:
            min_abs = upper_bin

        # now choose a new middle point for the upper and lower things
        lower_bin = min_abs + int(abs(max_abs-min_abs)/2)
        upper_bin = lower_bin + 1
    
    assert(x>domain[lower_bin] and x<=domain[upper_bin])
    if closest:
        return( lower_bin if abs(domain[lower_bin]-x)<abs(domain[upper_bin]-x) else upper_bin )
    else:
        return(lower_bin, upper_bin)


def get_closest(x, domain, mapped):
    """
    We imagine some function maps from "domain" to "mapped"

    We have several points evaluated for this function
        domain - list-like of floats. 
        mapped - list-like of floats. Entries in domain, evaluated by the function

    The user provides a value "x," and then we interpolate the mapped value on either side of 'x' to approximate the mapped value of 'x' 
    
    This is really just a linear interpolator 
    """
    if not isinstance(domain, (tuple,list,np.ndarray)):
        raise TypeError("'domain' has unrecognized type {}, try {}".format(type(domain), list))
    if not isinstance(mapped, (tuple,list,np.ndarray)):
        print(mapped)
        raise TypeError("'mapped' has unrecognized type {}, try {}".format(type(mapped), list))
    if not isinstance(x, (float,int)):
        raise TypeError("'x' should be number-like, not {}".format(type(x)))

    if len(domain)!=len(mapped):
        raise ValueError("'domain' and 'mapped' should have same length, got len(domain)={}, len(mapped)={}".format(len(domain), len(mapped)))
    
    lower_bin, upper_bin = get_loc(x, domain)
    
    # linear interp time
    x1 = domain[lower_bin]
    x2 = domain[upper_bin]
    y1 = mapped[lower_bin]
    y2 = mapped[upper_bin]

    slope = (y2-y1)/(x2-x1)
    value = (x*slope + y2 -x2*slope)

#    print("({}, {}) to ({}, {}) gave {}".format(x1,y1,x2,y2, value))
    
    return(value)

class Calculable:
    def __init__(self):
        if not os.path.exists(self.filename):
            self._obj = self.generate()
            self._save(self.obj)
        else:
            self._obj = self._load()

    @property
    def obj(self):
        return(self._obj)

    @property
    def filename(self):
        raise NotImplemented("Override default function!")
    
    def _save(self, obj):
        f=open(self.filename,'wb')
        pickle.dump(obj, f, -1)
        f.close()

    def _load(self):
        f = open(self.filename,'rb')
        all_data = pickle.load(f)
        f.close()
        return(all_data)

    def generate(self):
        raise NotImplemented("Override default function!")

    def eval(self, value):
        raise NotImplemented("Override default function!")


class KappaGrabba(Calculable):
    """
    This object calculates the smearing factor "kappa" for a bunch of possible angular errors, then saves them in a data file 
    """
    def __init__(self):
        self.numb = 1000

        self.czen_range = np.linspace(-0.99999,0.99999,self.numb)

        Calculable.__init__(self)
        
    @property
    def filename(self):
        return os.path.join(os.path.dirname(__file__), "resources", "kappa_file.pickle")

    def generate(self):
        """
        Should only be called once ever. This calculates all the kappa values! 
        """
        print("Generating Kappas for Angular uncertainty")
        kappas = np.zeros(shape=np.shape(self.czen_range))

        for i_ang in range(len(self.czen_range)):
            czenith =  self.czen_range[i_ang]
            def funct(kappa):
                if kappa<=0:
                    return(1e6)
                else:
                    return (exp(kappa) - exp(kappa*czenith))/(2*sinh(kappa)) - 0.5
            
            soln = opt.root(funct, np.array([10.]))
            kappas[i_ang] = soln.x[0]

        return(kappas)

    def eval(self, rad_error):
        """
        Accesses the stored calculated values for kappa
        """
        if not isinstance(rad_error, (int, float)):
            raise TypeError("Receied invalid datatype for rad error: {}".format(type(rad_error)))
        
        value = get_closest(cos(rad_error), self.czen_range, self.obj)
        return(value)


class bhist:
    """
    It's a 1D or 2D histogram! BHist is for "Ben Hist" or "Binned Hist" depending on who's asking. 

    I made this so I could have a binned histogram that could be used for adding more stuff at arbitrary places according to some "edges" it has. The object would handle figuring out which of its bins would hold the stuff. 

    Also made with the potential to store integers, floats, or whatever can be added together and has both an additive rule and some kind of identity element correlated with the default constructor. 
        If a non-dtype entry is given, it will be explicitly cast to the dtype. 
    """
    def __init__(self,edges, dtype=float):
        """
        Arg 'edges' should be a tuple of length 1 or 2. Length 1 for 1D hist, and length 2 for 2D hist. 
        These edges represent the bin edges. 

        The type-checking could use a bit of work... Right now for 1D histograms you need to give it a length-1 list. 
        """

        if not (isinstance(edges, list) or isinstance(edges, tuple) or isinstance(edges, np.ndarray)):
            raise TypeError("Arg 'edges' must be {}, got {}".format(list, type(edges)))


        for entry in edges:
            if not (isinstance(entry, list) or isinstance(entry, tuple) or isinstance(entry, np.ndarray)):
                raise TypeError("Each entry in 'edges' should be list-like, found {}".format(type(entry)))
            if len(entry)<2:
                raise ValueError("Entries in 'edges' must be at least length 2, got {}".format(len(entry)))
        
        self._edges = np.sort(edges) # each will now be increasing. I think this is Quicksort? 
        self._dtype = dtype

        # Ostensibly you can bin strings... not sure why you would, but you could! 
        try:
            x = dtype() + dtype()
        except Exception:
            raise TypeError("It appears impossible to add {} together.".format(dtype))

        # build the function needed to register additions to the histograms.
        dims = tuple([len(self._edges[i])-1 for i in range(len(self._edges))])
        self._fill = np.zeros( shape=dims, dtype=self._dtype )
        def register( amt, *args, density=True):
            """
            Tries to bin some data passed to the bhist. Arbitrarily dimensioned cause I was moving from 2D-3D and this seemed like a good opportunity 
                amt is the amount to add
                *args specifies the coordinates in our binned space 
                density specifies whether or not we want to divide out bin widths to make amt a density

            As a note, the density thing is implemented as-is since when using this, you won't know how wide the target bin is when you call this register function. You also can't just divide it out afterwards. This needs to happen between getting the bin location and adding it to the bin! 
            """
            if not len(args)==len(self._edges): 
                raise ValueError("Wrong number of args to register! Got {}, not {}".format(len(args), len(self._edges)))
            if not isinstance(amt, self._dtype):
                try:
                    amount = self._dtype(amt)
                except TypeError:
                    raise TypeError("Expected {}, got {}. Tried casting to {}, but failed.".format(self._dtype, type(amt), self._dtype))
            else:
                amount = amt

            bin_loc = tuple([get_loc( args[i], self._edges[i])[0] for i in range(len(args))]) # get the bin for each dimension

            # Verifies that nothing in the list is None-type
            if all([x is not None for x in bin_loc]):
                # itemset works like ( *bins, amount )
                if density: 
                    widths = self.widths 
                    for dim in range(len(self._edges)):
                        amount/=widths[dim][bin_loc[dim]]
                try:
                    self._fill.itemset(bin_loc, self._fill.item(tuple(bin_loc))+amount)
                except TypeError:
                    print("bin_loc: {}".format(bin_loc))
                    print("amount: {}".format(amount))
                    print("previous: {}".format(self._fill.item(tuple(bin_loc))))
                    import sys 
                    sys.exit()
                return tuple(bin_loc)
        self.register = register
 
    # some access properties. Note these aren't function calls. They are accessed like "object.centers" 
    @property
    def centers(self):
        complete = [ [0.5*(subedge[i+1]+subedge[i]) for i in range(len(subedge)-1)] for subedge in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def edges(self):
        complete = [[value for value in subedge] for subedge in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def widths(self):
        complete = [[abs(subedges[i+1]-subedges[i]) for i in range(len(subedges)-1)] for subedges in self._edges]
        return(complete[0] if len(self._edges)==1 else complete)
    @property
    def fill(self):
        return(self._fill)