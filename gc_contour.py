""" gc_contour.py

Created on Fri Jan 21 9:42:00 2022
@author: Shin Satoh

Description:
This program is intended to calculate the angles between
the surface grids on Europa and the local B field lines.

"""

# %% ライブラリのインポート
from numba import jit, f8
from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
import time
from multiprocessing import Pool

# FAVORITE COLORS (FAVOURITE COLOURS?)
color = ['#6667AB', '#0F4C81', '#5B6770', '#FF6F61', '#645394',
         '#84BD00', '#F6BE00', '#F7CAC9', '#16137E', '#45B8AC']


richtext = input('rich text (y) or (n): ')
if richtext == 'y':
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath \usepackage{siunitx} \sisetup{detect-all}'
    #    \usepackage{helvet}     # helvetica font
    #    \usepackage{sansmath}   # math-font matching  helvetica
    #    \sansmath               # actually tell tex to use it!
    #    \usepackage{siunitx}    # micro symbols
    #    \sisetup{detect-all}    # force siunitx to use the fonts


#
#
# %% CONSTANTS
RJ = float(7E+7)        # Jupiter半径   単位: m
mJ = float(1.90E+27)    # Jupiter質量   単位: kg
RE = float(1.56E+6)     # Europa半径    単位: m
mE = float(4.8E+22)     # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C

G = float(6.67E-11)     # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
omgJ = float(1.74E-4)   # 木星の自転角速度 単位: rad/s
omgE = float(2.05E-5)   # Europaの公転角速度 単位: rad/s
omgR = omgJ-omgE        # 木星のEuropaに対する相対的な自転角速度 単位: rad/s
eomg = np.array([-np.sin(np.radians(10.)),
                 0., np.cos(np.radians(10.))])
omgRvec = omgR*eomg
# omgR2 = omgR
# omgR2 = 0.5*omgR        # 0.5*omgR = 51500 m/s (at Europa's orbit)
omgR2 = 0.1*omgR        # 0.1*omgR = 10300 m/s (at Europa's orbit)
# omgR2 = 0.02*omgR        # 0.02*omgR = 2060 m/s (at Europa's orbit)
# omgR2 = 0.01*omgR        # 0.02*omgR = 1030 m/s (at Europa's orbit)
omgR2vec = omgR2*eomg


#
#
# %% 途中計算でよく出てくる定数の比
# A1 = float(e/me)                  # 運動方程式内の定数
# A2 = float(mu*Mdip/4/3.14)        # ダイポール磁場表式内の定数
A1 = float(-1.7582E+11)             # 運動方程式内の定数
A2 = 1.60432E+20                    # ダイポール磁場表式内の定数
A3 = 4*3.1415*me/(mu*Mdip*e)        # ドリフト速度の係数


#
#
# %% EUROPA POSITION (DETERMINED BY MAGNETIC LATITUDE)
lam = 10.0
L96 = 9.6*RJ  # Europa公転軌道 L値

# 木星とtrace座標系原点の距離(x軸の定義)
# Europaの中心を通る磁力線の脚(磁気赤道面)
R0 = L96*(np.cos(np.radians(lam)))**(-2)
R0x = R0
R0y = 0
R0z = 0
R0vec = np.array([R0x, R0y, R0z])

# 初期条件座標エリアの範囲(木星磁気圏動径方向 最大と最小 phiJ=0で決める)
r_ip = (L96+1.15*RE)*(math.cos(math.radians(lam)))**(-2)
r_im = (L96-1.15*RE)*(math.cos(math.radians(lam)))**(-2)

# Europaのtrace座標系における位置
eurx = L96*math.cos(math.radians(lam)) - R0x
eury = 0 - R0y
eurz = L96*math.sin(math.radians(lam)) - R0z

# 遠方しきい値(z方向) 磁気緯度で設定
z_p_rad = math.radians(11.0)      # 北側
z_p = R0*math.cos(z_p_rad)**2 * math.sin(z_p_rad)
z_m_rad = math.radians(2.0)      # 南側
z_m = -R0*math.cos(z_m_rad)**2 * math.sin(z_m_rad)


#
#
# %% 高速な内積計算
@jit(nopython=True, fastmath=True)
def vecdot(vec1, vec2):
    """
    DESCRIPTION IS HERE.
    """
    dot = vec1[0, :, :]*vec2[0, :, :] + vec1[1, :, :] * \
        vec2[1, :, :] + vec1[2, :, :]*vec2[2, :, :]

    return dot


#
#
# %% 任意の軸回りのベクトル回転
@jit('f8[:](f8[:],f8)', nopython=True, fastmath=True)
def Corotation(Rvec, theta):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル \\
    `theta` ... 共回転の回転角 [RADIANS]
    """
    n1 = eomg[0]
    n2 = eomg[1]
    n3 = eomg[2]

    cos = math.cos(theta)
    sin = math.sin(theta)

    Rmatrix = np.array([
        [(n1**2)*(1-cos)+cos, n1*n2*(1-cos)-n3*sin, n1*n3*(1-cos)+n2*sin],
        [n1*n2*(1-cos)+n3*sin, (n2**2)*(1-cos)+cos, n2*n3*(1-cos)-n2*sin],
        [n1*n3*(1-cos)-n2*sin, n2*n3*(1-cos)+n1*sin, (n3**2)*(1-cos)+cos]
    ])

    Rvec_new = np.array([
        Rmatrix[0, 0]*Rvec[0] + Rmatrix[0, 1]*Rvec[1] + Rmatrix[0, 2]*Rvec[2],
        Rmatrix[1, 0]*Rvec[0] + Rmatrix[1, 1]*Rvec[1] + Rmatrix[1, 2]*Rvec[2],
        Rmatrix[2, 0]*Rvec[0] + Rmatrix[2, 1]*Rvec[1] + Rmatrix[2, 2]*Rvec[2],
    ])

    return Rvec_new


#
#
# %% 磁場
def Bfield(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    x = Rvec[0, :, :]
    y = Rvec[1, :, :]
    z = Rvec[2, :, :]

    print('ok')

    # distance
    R2 = x**2 + y**2
    r_5 = np.sqrt(R2 + z**2)**(-5)

    # Magnetic field
    Bvec = A2*r_5*np.array([3*z*x, 3*z*y, 2*z**2 - R2])

    return Bvec


#
#
# %%
def Babs(Rvec):
    """
    `Rvec` ... <ndarray> ダイポール原点の位置ベクトル
    """
    # x, y, zは木星からの距離
    Bvec = Bfield(Rvec)
    B = np.sqrt(Bvec[0, :, :]**2 + Bvec[1, :, :]**2 + Bvec[2, :, :]**2)

    return B


#
#
# %%
def ax0(H, X, Y):
    # x軸ラベル(標準的なwest longitude)
    xticklabels = ['360$^\\circ$W', '270$^\\circ$W',
                   '180$^\\circ$W', '90$^\\circ$W', '0$^\\circ$']
    yticklabels = ['90$^\\circ$N', '45$^\\circ$N',
                   '0$^\\circ$', '45$^\\circ$S', '90$^\\circ$S']

    # 図のタイトル
    title = 'Latitude-West Longitude, $\\lambda=$' + \
        str(lam) + '$^\\circ$'

    # 描画
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    # ax.set_xticks(np.linspace(-180, 180, 5))
    # ax.set_yticks(np.linspace(0, 180, 5))
    # ax.set_xticklabels(xticklabels, fontsize=12)
    # ax.set_yticklabels(yticklabels, fontsize=12)
    ax.invert_yaxis()
    cs = ax.contour(X, Y, H, levels=15, colors='k')
    ax.clabel(cs)
    """
    mappable0 = ax.pcolormesh(X, Y, H, cmap='magma',
                              vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical')
    pp.set_label('Precipitation Flux [cm$^{-2}$ s$^{-1}$]', fontsize=12)
    pp.ax.tick_params(labelsize=12)
    pp.ax.yaxis.get_offset_text().set_fontsize(12)
    """

    fig.tight_layout()
    fig.savefig('gc_contour_lam10.png', transparent=True)
    plt.show()

    return 0


#
#
# %%
def main():
    # 表面緯度経度
    s_colat0 = np.radians(np.linspace(0, 180, 100))
    s_long0 = np.radians(np.linspace(-180, 180, 2*s_colat0.size))

    # メッシュに
    s_long, s_colat = np.meshgrid(s_long0, s_colat0)

    # 表面法線ベクトル
    nvec = np.array([
        np.sin(s_colat)*np.cos(s_long),
        np.sin(s_colat)*np.sin(s_long),
        np.cos(s_colat)
    ])
    print(nvec.shape)

    # 法線ベクトルの回転
    x_rot = nvec[0, :, :]*math.cos(math.radians(-lam)) + \
        nvec[2, :, :]*math.sin(math.radians(-lam))
    y_rot = nvec[1, :, :]
    z_rot = -nvec[0, :, :]*math.sin(math.radians(-lam)) + \
        nvec[2, :, :]*math.cos(math.radians(-lam))
    nvec[0, :, :] = x_rot
    nvec[1, :, :] = y_rot
    nvec[2, :, :] = z_rot
    print(nvec.shape)

    # Trace座標系に
    Rinitvec = RE*nvec
    Rinitvec[0, :, :] += eurx
    Rinitvec[1, :, :] += eury
    Rinitvec[2, :, :] += eurz

    # 単位磁場ベクトル
    R0vec_array = np.ones(nvec.shape)
    R0vec_array[0, :, :] = R0x*R0vec_array[0, :, :]
    R0vec_array[1, :, :] = R0y*R0vec_array[1, :, :]
    R0vec_array[2, :, :] = R0z*R0vec_array[2, :, :]
    print(R0vec_array.shape)

    B = Babs(Rinitvec + R0vec_array)
    print(B.shape)
    B_array = np.ones(nvec.shape)
    B_array[0, :, :] = B*B_array[0, :, :]
    B_array[1, :, :] = B*B_array[1, :, :]
    B_array[2, :, :] = B*B_array[2, :, :]
    bvec = Bfield(Rinitvec + R0vec_array)/B_array

    # 表面法線ベクトルと単位磁場ベクトルの内積
    d = vecdot(nvec, bvec)
    print(np.max(d))
    print(np.min(d))

    # 角度[degrees]に変換
    arg = np.degrees(np.arccos(d))
    print(np.max(arg))
    print(np.min(arg))

    # 描画用メッシュ
    yedges = np.linspace(0, 180, 100)
    xedges = np.linspace(-180, 180, 2*yedges.size)
    X, Y = np.meshgrid(xedges, yedges)

    ax0(arg, X, Y)

    return 0


#
#
# %%
if __name__ == '__main__':
    a = main()
