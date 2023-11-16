
from pone_newphs.fitter import Fitter
from pone_newphs import utils
import numpy as np

import json

param = utils.NSIParam()
test = Fitter(param)

evaled = test.evaluate_exp(np.zeros(test.dim))

fc = utils.FILECLASS.REAL
root_folder = utils.make_root(fc)
outfile_name = utils.get_filename(root_folder, "realization.json", param)

out = {"data":evaled.tolist()}

_obj = open(outfile_name, 'wt')
json.dump(out, _obj, indent=4)
_obj.close()
print("saved ",outfile_name)

