
import numpy as np

with open('./data_loaders/prt_val_f.txt', 'w') as f:
    info = list(f.read())
ids = np.load('./data_loaders/brain3D_sample_label_age_id.npy')
i = ids[0].decode("utf-8")

print()



