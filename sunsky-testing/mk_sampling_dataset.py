import sys; sys.path.insert(0, "build/python")

import numpy as np
import pandas as pd

import mitsuba as mi

mi.set_variant("llvm_rgb")

filename = "sunsky-testing/res/datasets/model_hosek.csv"
destination_folder = "sunsky-testing/res/datasets/"

df = pd.read_csv(filename)
df.pop('RMSE')
df.pop('MAE')
df.pop('Volume')
df.pop('Normalization')
df.pop('Azimuth')

arr = df.to_numpy()

sort_args = np.lexsort([arr[::, 1], arr[::, 0]])
simplified_arr = arr[sort_args, 2:]
simplified_arr[::, 1] = np.pi/2 - simplified_arr[::, 1]

shape = (9, 30, 5, 5)
mi.array_to_file(f"{destination_folder}/tgmm_tables.bin", mi.Float(np.ravel(simplified_arr)), shape)
