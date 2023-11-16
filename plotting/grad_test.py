import matplotlib.pyplot as plt 
import numpy as np 
from  pone_newphs.fitter import Fitter
from pone_newphs import utils

test = Fitter(utils.NSIParam())

n_param = test.dim 

one_sig =np.diag(test.std)
nominal = test.evaluate_exp(np.zeros(n_param))

names = test.names

for i in range(len(one_sig)):
    print("on {}".format(names[i]))
    effect = test.evaluate_exp(one_sig[i])

    plt.clf()
    plt.pcolormesh(test.z_bins, test.e_bins, np.transpose((effect-nominal)/nominal), vmin=-0.05, vmax=0.05, cmap="RdBu")
    plt.xlabel("Cos theta",size=14)
    plt.yscale('log')
    plt.colorbar()
    plt.ylabel("Energy [GeV]",size=14)

    plt.savefig("./plots/grad_{}.png".format(names[i]), dpi=400)
