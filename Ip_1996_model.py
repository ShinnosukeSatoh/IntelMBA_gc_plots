""" Ip_1996_model.py

Created on Wed Feb 2 2022
@author: Shin Satoh

"""


# %% ライブラリのインポート
# from numba import jit
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.ticker as mticker
# from matplotlib import rc
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
# import time

# from numpy.lib.npyio import savez_compressed
# from multiprocessing import Pool

# from numpy.lib.function_base import _flip_dispatcher
color = ['#0F4C81', '#FF6F61', '#645394', '#84BD00', '#F6BE00', '#F7CAC9']

# matplotlib フォント設定
"""
plt.rcParams.update({'font.sans-serif': "Arial",
                     'font.family': "sans-serif",
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     })
"""

richtext = input('rich text (y) or (n): ')
if richtext == 'y':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'
    #    \usepackage{helvet}     # helvetica font
    #    \usepackage{sansmath}   # math-font matching  helvetica
    #    \sansmath               # actually tell tex to use it!
    #    \usepackage{siunitx}    # micro symbols
    #    \sisetup{detect-all}    # force siunitx to use the fonts

RE = 1.5E+6
x_array = np.linspace(-5*RE, 5*RE, 1000)
y_array = x_array
xmesh, ymesh = np.meshgrid(x_array, y_array)

a = 0.75
V0 = 76*1E+3  # m/s
Rc = RE

r = np.sqrt(xmesh**2 + ymesh**2)
Vx = -2*(1-a)*V0*((Rc/r)**2)*(xmesh*ymesh)/(r**2)
Vy = V0 + (1-a)*V0*((Rc/r)**2)*(1-2*(ymesh**2)/(r**2))
Vx[np.where(r < RE)] = 0
Vy[np.where(r < RE)] = a*V0
Vy += 30*1E+3  # m/s

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title('$|$V$|$, $\\alpha=$ '+str(a), fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
mappable0 = ax.pcolormesh(xmesh/RE, ymesh/RE,
                          np.sqrt(Vx**2 + Vy**2)/1000, cmap='magma', vmin=0, vmax=200, shading='auto')
plt.quiver(xmesh[::40, ::40]/RE, ymesh[::40, ::40]/RE, Vx[::40, ::40]/5000, Vy[::40,
           ::40]/5000, color='k', angles='xy', scale_units='xy', scale=100)  # ベクトル場をプロット
pp = fig.colorbar(mappable0, orientation='vertical')
pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
pp.ax.tick_params(labelsize=12)
pp.ax.yaxis.get_offset_text().set_fontsize(12)
plt.show()
print(np.max(np.sqrt(Vx**2 + Vy**2))/1000)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title('Vx, $\\alpha=$ '+str(a), fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
mappable0 = ax.pcolormesh(xmesh/RE, ymesh/RE, Vx/1000,
                          vmin=-120, vmax=120, cmap='magma', shading='auto')
pp = fig.colorbar(mappable0, orientation='vertical')
pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
pp.ax.tick_params(labelsize=12)
pp.ax.yaxis.get_offset_text().set_fontsize(12)
plt.show()
print(np.max(Vx)/1000)

fig, ax = plt.subplots(figsize=(6, 5))
ax.set_title('Vy, $\\alpha=$ '+str(a), fontsize=12)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
mappable0 = ax.pcolormesh(xmesh/RE, ymesh/RE, Vy/1000,
                          vmin=-120, vmax=120, cmap='magma', shading='auto')
pp = fig.colorbar(mappable0, orientation='vertical')
pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
pp.ax.tick_params(labelsize=12)
pp.ax.yaxis.get_offset_text().set_fontsize(12)
plt.show()
print(np.min(Vy)/1000)
