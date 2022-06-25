
# Make posible importing modules from parent directory
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import matplotlib.pyplot as plt
from cloudatlas.utils import DataFeeder, animate_time_series
from rich import print
test_data = DataFeeder("splitted_dataset/test_splitted_data").feed()

a, b, c = next(test_data)

print(f"{a.shape} , {b.shape} , {c.shape}")

u = animate_time_series(b[16])
plt.show()