import pandas as pd
import numpy as np
import os,re

dta = pd.read_csv('/Users/pengfeiwang/Desktop/dogkfc/Project3_poodleKFC_train/sift_features.csv')
dta = pd.DataFrame.transpose(dta)
def Obj_Type(x):
    if str(x).find('chicken') != -1: return('1')
    else: return(0)
    
dta['dtype'] = list(dta.index)
dta['dtype'] = dta['dtype'].map(lambda x: Obj_Type(x))
