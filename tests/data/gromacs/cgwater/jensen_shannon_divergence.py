import numpy as np
import math


CG_pairs=['LJ-LJ']#I1-I1','I1-I2','I1-I3','I1-CT','I1-PF','I2-I2','I2-I3','I2-CT','I2-PF','I3-I3','I3-CT','I3-PF','CT-CT','CT-PF','PF-PF']

diverge=[]

for p in CG_pairs:
    AA=np.loadtxt('/data/isilon/woerner/ionic_liquids/CG_T/multi_T/smoothing/test_water/AA/rdf_'+p+'.xvg')
    CG=np.loadtxt('rdf_'+p+'.xvg')

    KL=0.0

    for i in range(301):
        if (AA[i,1]!=0.0):
            if (CG[i,1]!=0.0):
                m=0.5*(AA[i,1]+CG[i,1])
                k=0.5*(CG[i,1]*math.log10(CG[i,1]/m))+0.5*(AA[i,1]*math.log10(AA[i,1]/m))
                KL=KL+k




    diverge.append(KL)
kldiv=np.asarray(diverge)
np.savetxt('js_divergence.dat',kldiv)
