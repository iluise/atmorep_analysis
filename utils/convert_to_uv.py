####################################################################################################
#
#  Copyright (C) 2023
#
####################################################################################################
#
#  project     : atmorep
#
#  author      : atmorep collaboration
# 
#  description : script to convert from u-v wind components to vorticity and divergence. 
#
#  license     : MIT licence
#                
####################################################################################################


from spharm import Spharmt
import numpy as np

vort = np.fromfile( 'vort.dat').astype( np.float32).reshape( 721, 1440)
div = np.fromfile( 'div.dat').astype( np.float32).reshape( 721, 1440)

spharm = Spharmt( 1440, 721, legfunc='stored')
sh_div = spharm.grdtospec( div )
sh_vort = spharm.grdtospec( vort )
u,v = spharm.getuv( sh_vort, sh_div)

u.tofile( 'velocity_u.dat')
v.tofile( 'velocity_v.dat')

print( 'Finished.')
