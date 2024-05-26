# data at https://zenodo.org/record/3606767/#.Y87pt3bMJPa
from glob import glob
import numpy as np
import pandas as pd

import src.utils as utils 

def load_data(cvx_dim, multi_clf='', 
              upper_mass_cut:int=450, training:bool=True):
    
    output, _ = load_multi_cls(multi_clf, upper_mass_cut=upper_mass_cut)
    
    if cvx_dim==1: # binary decorrelation
        
            output = {key: item for key, item in output.items()}
            
    if training:
        bkg_mask = output["labels"]==0
        output = {i: output[i][np.ravel(bkg_mask)] for i in output}

    return output

        
def load_multi_cls(path, key=None, upper_mass_cut=450):
        
    data = utils.load_hdf(path)
    key = list(data.keys())[0] if key is None else key
    if len(list(data.keys()))>1:
        raise KeyError("multiple keys in data")
    data = data[key][:]
    data = data[(data[:,0]>=20) & (data[:,0]<upper_mass_cut)]
    data = data[:,:3]
    columns = ["mass", "label", "w_score"]
    data = pd.DataFrame(data, columns=columns)
    output = {"mass": data["mass"].to_numpy(), "labels": data["label"].to_numpy(),
              "encodings": data["w_score"].to_numpy()}
    return output, data
