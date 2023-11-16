from area_weighter import bin_cth, bin_e, AreaWeighter
import numpy as np
import matplotlib.pyplot as plt 


aw = AreaWeighter()

cths = [-1.0,  0.0, 1.0]
true_e = 10**bin_e
for cth in cths:
    vals = aw(cth, bin_e)[0]
    

    plt.plot(true_e, vals, label="Cth {}".format(cth))

plt.xscale('log')
plt.xlabel("Energy [GeV]", size=16)
plt.ylabel("Area [m2]")
plt.yscale('log')
plt.legend()
plt.show()