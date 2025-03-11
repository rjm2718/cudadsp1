#!/bin/env python3

# read pairs of numbers from stdin, plot it

import sys, math
import numpy as np
import matplotlib.pyplot as plt

filenames = sys.argv[1:]
if len(filenames) < 1:
    sys.exit(0)

plt.figure(figsize=(20, len(filenames)*6))

p = 1
for fn in filenames:

    A=[]
    for l in open(fn).readlines():
        (r,c) = tuple(map(float, l.strip().split()))
        A.append( math.sqrt(r*r+c*c) )

    plt.subplot(len(filenames),1,p) 
    plt.plot(np.array(A))
    plt.title(fn)
    p+=1


plt.tight_layout()
plt.show()
