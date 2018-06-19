import yaml
import numpy as np

with open('calibration.yaml') as f:
    loadeddict = yaml.load(f)
camera_matrix = loadeddict.get('camera_matrix')
dist_coeffs = loadeddict.get('dist_coeff')

c_m = np.asarray(camera_matrix)

print(c_m)