#!/bin/env python3

# read pairs of numbers from stdin, plot it

import sys, math
import numpy as np
import matplotlib.pyplot as plt

A=[]
for l in sys.stdin:
    (r,c) = tuple(map(float, l.strip().split()))
    A.append( math.sqrt(r*r+c*c) )

a = np.array(A)
plt.figure(figsize=(20, 12))
plt.plot(a)
plt.show()
