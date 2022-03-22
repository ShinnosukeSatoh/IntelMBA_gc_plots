""" gc_305_plot.py

Created on Thu Mar 22 2022
@author: Shin Satoh

"""


# %% ライブラリのインポート
# from statistics import mode
# from numba import jit
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.ticker as ptick
# from matplotlib.colors import LinearSegmentedColormap  # colormapをカスタマイズする
# from matplotlib import rc
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
# import time

# from numpy.lib.npyio import savez_compressed
# from multiprocessing import Pool

# from numpy.lib.function_base import _flip_dispatcher
color = ['#0F4C81', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#0F80E6']


# matplotlib フォント設定
plt.rcParams.update({'font.sans-serif': "Arial",
                     'font.family': "sans-serif",
                     'mathtext.fontset': 'custom',
                     'mathtext.rm': 'Arial',
                     'mathtext.it': 'Arial:italic',
                     'mathtext.bf': 'Arial:italic:bold'
                     })


#
#
# %% 磁場
A2 = 1.60432E+20


def Bfield(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    x = Rvec[0]
    y = Rvec[1]
    z = Rvec[2]

    # distance
    R2 = x**2 + y**2
    r_5 = math.sqrt(R2 + z**2)**(-5)

    # Magnetic field
    Bvec = A2*r_5*np.array([3*z*x, 3*z*y, 2*z**2 - R2])

    return Bvec


#
#
# %% dipole field line equation
def dipole(x, req):
    z = np.sqrt((req*x**2)**(2/3) - x**2)
    return z


#
#
# %%
def xz_rot(x, z, theta):
    xnew = x*np.cos(np.radians(theta)) + z*np.sin(np.radians(theta))
    znew = -x*np.sin(np.radians(theta)) + z*np.cos(np.radians(theta))
    return xnew, znew


RJ = 1
RE = (1.5E+6)/(7E+7)
req = 9.4*RJ
lam = -10

# ダイポール軸
x_axis = np.linspace(-2*RJ, 2*RJ, 3)*np.sin(np.radians(0))
z_axis = np.linspace(-2*RJ, 2*RJ, 3)*np.cos(np.radians(0))
x_axis, z_axis = xz_rot(x_axis, z_axis, lam)

# 磁気赤道面
xeq = np.linspace(-1.25*RJ, 10*RJ, 3)
zeq = xeq*0
xeq, zeq = xz_rot(xeq, zeq, lam)

# Europaの位置
xe = 9.4*RJ
ze = 0
xer, zer = xz_rot(xe, ze, -lam)
req_e = ((xer**2 + zer**2)**(3/2)) / (xer**2)

# Europaを貫く磁力線
x_new = np.linspace(0, req_e+500, 1000000)
z_new1 = dipole(x_new, req_e)
z_new2 = -dipole(x_new, req_e)
x_new2, z_new1 = xz_rot(x_new, z_new1, lam)
x_new22, z_new2 = xz_rot(x_new, z_new2, lam)

fig, ax = plt.subplots(dpi=120)
ax.set_axisbelow(True)
ax.set_title('Magnetic Lat. '+str(lam)+'$^\\circ$ ($L$ = ' +
             '{:.2f}'.format(req_e/RJ)+')',
             fontsize=25,
             weight='bold')
ax.set_xlabel('$x$ [R$_{\\rm J}$]', fontsize=25)
ax.set_ylabel('$z$ [R$_{\\rm J}$]', fontsize=25)
ax.tick_params(axis='x', labelsize=25)
ax.tick_params(axis='y', labelsize=25)
ax.plot(x_new2, z_new1, linewidth=1.5,
        color='#000000', zorder=2)
ax.plot(x_new22, z_new2, linewidth=1.5, color='#000000')
ax.plot(x_axis, z_axis, linewidth=1.5, linestyle='dashed', color='#000000')
ax.plot(xeq, zeq, linewidth=1.5, linestyle='dashed', color='#000000')
ax.add_patch(plt.Circle((xe, ze), 16*RE, color=color[6], zorder=3))
ax.add_patch(plt.Circle((0, 0), RJ, color=color[6], zorder=3))
ax.set_aspect('equal')

# plt.legend()
fig.tight_layout()

plt.show()
