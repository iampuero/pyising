from src.inference import inference
import numpy as np
from scipy.sparse import load_npz

raster = load_npz('data/degus_0_nmovie_dt_0.02.npz')

inference(raster, "results", "degus_0_nmovie_dt_0.02", 0.1, 0.1)