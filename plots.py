import matplotlib.pyplot as plt
import pandas as pd
import os.path as osp

if osp.isfile("mkeys_per_sec.csv"):
    mkeys = pd.read_csv("mkeys_per_sec.csv")

    sizes = mkeys["array size"]
    thrust_radix_ks = mkeys["thrust radix mkeys"]
    thrust_compare_ks = mkeys["thrust compare mkeys"]
    merge_ks = mkeys["merge mkeys"]
    bitonic_ks = mkeys["bitonic mkeys"]

    fig, ax = plt.subplots()
    ax.plot(sizes, thrust_radix_ks, color="blue", label="Thrust Radix", marker="o")
    ax.plot(sizes, thrust_compare_ks, color="purple", label="Thrust Compare", marker="o")
    ax.plot(sizes, merge_ks, label="Merge", color="green", marker="o")
    ax.plot(sizes, bitonic_ks, label="Bitonic", color="red", marker="o")
    ax.set_xscale("log", base=2)
    ax.set_title("Work Throughput Plot")
    ax.legend()
    ax.set_xlabel("Array Size (Power of 2)")
    ax.set_ylabel("MKeys / Second")
    plt.show()

if osp.isfile("gbps.csv"):
    mkeys = pd.read_csv("gbps.csv")

    sizes = mkeys["array size"]
    thrust_radix_ks = mkeys["thrust radix gbps"]
    thrust_compare_ks = mkeys["thrust compare gbps"]
    merge_ks = mkeys["merge gbps"]
    bitonic_ks = mkeys["bitonic gbps"]

    fig, ax = plt.subplots()
    ax.plot(sizes, thrust_radix_ks, color="blue", label="Thrust Radix", marker="o")
    ax.plot(sizes, thrust_compare_ks, color="purple", label="Thrust Compare", marker="o")
    ax.plot(sizes, merge_ks, label="Merge", color="green", marker="o")
    ax.plot(sizes, bitonic_ks, label="Bitonic", color="red", marker="o")
    ax.set_xscale("log", base=2)
    ax.set_title("Memory Bandwidth Plot")
    ax.legend()
    ax.set_xlabel("Array Size (Power of 2)")
    ax.set_ylabel("GBPS")
    plt.show()
