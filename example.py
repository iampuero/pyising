#from src import dataDrivenIsing as ddi
import numpy as np
from scipy.sparse import load_npz

raster = load_npz('data/degus_0_nmovie_dt_0.02.npz')

print(raster.shape)