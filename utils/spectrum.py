
import numpy as np
import glob
import json
import os
import code
# code.interact(local=locals())

from spharm import Spharmt
# import pyshtools as pysh


lmax = 720

spharm = Spharmt( 1440, 721, legfunc='stored')
# spectrum = pysh.spectralanalysis.spectrum( sh)  # <-> u^2
# idxs_l = [ for l in range(lmax)]

spectrum_mean = np.zeros( lmax+1)

num_files = 0
for tidx in range( 23) :
  
  fname  = sorted(glob.glob( 'id2nsb5q1d_preds*tidx{}*'.format(tidx)))
  if not 1 == len(fname) :
    continue
  print( 'Processing {}.'.format( fname[0]))

  preds = np.fromfile( fname[0], dtype=np.float32)
  preds = preds.reshape( [5,3,721,1440])

  sh = spharm.grdtospec( preds[0,0], ntrunc=lmax)

  idx = 0
  data = [[] for l in range(lmax+1)]
  for m in range( lmax + 1) :
    for l in range( m, lmax + 1) :
      data[l].append( sh[idx])
      idx += 1

  spectrum = np.zeros( lmax+1)
  for l in range(lmax+1) :
    temp = np.array( data[l], dtype=np.complex64)
    spectrum[l] = np.linalg.norm( temp)

  spectrum_mean += spectrum
  num_files += 1

spectrum_mean /= num_files

spectrum.tofile( 'spectrum.dat')



