#!/usr/bin/python3

# TODO: Standardize all string consts
## Devices:
class DEVICES():
    CPU = 'CPU'
    GPU = 'GPU'
    ALL = [ CPU, GPU ]

## Formats:
class FORMATS():
    DENSE_CROUT = 'Dense Crout [CPU]'
    DENSE_CROUT_ITERATIVE = 'Dense Iterative Crout [GPU] (16 threads)'
    BASELINE = DENSE_CROUT
    BEST = 'Best'