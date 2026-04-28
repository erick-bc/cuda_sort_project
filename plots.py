import numpy as np
import matplotlib.pyplot as plt

sizes = np.array([2**15, 2**20, 2**25, 2**27, 2**28])
# thrust_radix_ks = np.array([26.8, 6522.29, 16484.71, 17372.03, 17566.44])
# thrust_compare_ks = np.array([507.94, 3893.54, 8810.97, 13198.27, 14712.24])
# merge_ks = np.array([103.56, 1499.41, 1865.85, 1837.54, 1761.84])
# bitonic_ks = np.array([107.38, 743.65, 319.94, 282.7, 266.08]) 

thrust_radix_ks = np.array([16.73, 311.33, 630.63, 650.04, 650.86])
thrust_compare_ks = np.array([16.62, 317.51, 629.81, 650.04, 652.05])
merge_ks = np.array([13.62, 132.09, 206.46, 214.67, 217.05])
bitonic_ks = np.array([50.06, 538.97, 678.88, 680.42, 679.61])

fig, ax = plt.subplots()
ax.plot(sizes, thrust_radix_ks, label="Thrust Radix", marker="o")
ax.plot(sizes, thrust_compare_ks, label="Thrust Compare", marker="o")
ax.plot(sizes, merge_ks, label="Merge", marker="o")
ax.plot(sizes, bitonic_ks, label="Bitonic", marker="o")
ax.tick_params(axis="x", labelrotation=90)
ax.set_xscale("log", base=2)
# ax.set_title("Work Throughput Plot")
ax.set_title("Memory Bandwidth Plot")
ax.legend()
ax.set_xlabel("Array Size (Power of 2)")
ax.set_ylabel("GB / Second")
plt.show()
