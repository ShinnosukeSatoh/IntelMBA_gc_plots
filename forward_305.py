""" forward_305.py """
# <実行時間>
#
# <説明>
# ・着地点の座標(x,y,z)を入力
# ・着地点のEuropaローカル緯度経度を計算
# ・表面降り込みのマップを作成


# %% ライブラリのインポート
from numba import jit
# from numba.experimental import jitclass
import numpy as np
import math
import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from mpl_toolkits.mplot3d import Axes3D
# import time
# from multiprocessing import Pool

# from numpy.lib.function_base import _flip_dispatcher
color = ['#0F4C81', '#FF6F61', '#645394', '#84BD00', '#F6BE00', '#F7CAC9']


# %% FORWARD OR BACKWARD
FORWARD_BACKWARD = 1  # 1=FORWARD, -1=BACKWARD


# %% 定数
RJ = float(7E+7)        # Jupiter半径   単位: m
MJ = float(1.90E+27)    # Jupiter質量   単位: kg
REU = float(1.56E+6)    # Europa半径    単位: m
MEU = float(4.8E+22)    # Europa質量    単位: kg

c = float(3E+8)         # 真空中光速    単位: m/s
me = float(9.1E-31)     # 電子質量      単位: kg
e = float(-1.6E-19)     # 電子電荷      単位: C

Gc = float(6.67E-11)    # 万有引力定数  単位: m^3 kg^-1 s^-2

mu = float(1.26E-6)     # 真空中透磁率  単位: N A^-2 = kg m s^-2 A^-2
Mdip = float(1.6E+27)   # Jupiterのダイポールモーメント 単位: A m^2
# omega = FORWARD_BACKWARD*float(1.74E-4)  # 木星の自転角速度 単位: rad/s
omegaJ = FORWARD_BACKWARD*float(1.74E-4)    # 木星の自転角速度 単位: rad/s
omegaEUR = FORWARD_BACKWARD*float(2.05E-5)  # Europaの公転角速度 単位: rad/s
omegaRELATIVE = omegaJ-omegaEUR             # 木星のEuropaに対する相対的な自転角速度 単位: rad/s


# %% 途中計算でよく出てくる定数の比など
# A1 = float(e/me)             # 運動方程式内の定数
# A2 = float(mu*Mdip/4/3.14)  # ダイポール磁場表式内の定数
A1 = float(-1.7582E+11)    # 運動方程式内の定数
A2 = FORWARD_BACKWARD*1.60432E+20            # ダイポール磁場表式内の定数


# %% Europa Position
lam = 6.
eurReq = float(6.7E+8)  # 木星からEuropa公転軌道までの距離


# %% 木星からEuropaまでの距離
reur = eurReq

# 木星とtrace座標系原点の距離(x軸の定義)
R0 = reur*(math.cos(math.radians(lam)))**(-2)

# 初期条件座標エリアの範囲(最大と最小)
x_ip = (reur+1.025*REU)*(math.cos(math.radians(lam)))**(-2) - R0
x_im = (reur-1.025*REU)*(math.cos(math.radians(lam)))**(-2) - R0

# Europaのtrace座標系における位置
eurx = reur*math.cos(math.radians(lam)) - R0
eury = 17*REU   # 読み込むデータに応じて変更
eurz = reur*math.sin(math.radians(lam))


# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
def rot_dipole(xyz, lam):
    xi = xyz[:, 0]
    xj = xyz[:, 1]
    xk = xyz[:, 2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.stack([xrot, yrot, zrot], 1)
    return rot


# %% 座標変換(Europa軌道座標系から木星dipole座標系に)
def rot_dipole2(xyz, lam):
    xi = xyz[0]
    xj = xyz[1]
    xk = xyz[2]

    xrot = xi*np.cos(np.radians(lam))+xk*np.sin(np.radians(lam))
    yrot = xj
    zrot = -xi*np.sin(np.radians(lam))+xk*np.cos(np.radians(lam))

    rot = np.stack([xrot, yrot, zrot])
    return rot


# %% 着地点の余緯度と経度を調べる
def prefind(position):
    # 電子の座標
    # x... anti jovian
    # y... corotation
    # z... northern

    x = position[:, 0]
    y = position[:, 1]
    z = position[:, 2]
    theta = np.arccos(z/np.sqrt(x**2+y**2+z**2))
    phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2))
    print(np.degrees(np.stack([phi, theta], 1)))
    return np.stack([phi, theta], 1)


# %% マップ作成
def mapplot(maparray):
    # start = time.time()  # 時間計測開始
    # print(maparray)
    maparray = np.degrees(maparray)
    # print(maparray)
    xedges = list(np.linspace(-180, 180, 180))
    yedges = list(np.linspace(0, 180, int(len(xedges)/2)))
    H, xedges, yedges = np.histogram2d(
        maparray[:, 0], maparray[:, 1], bins=(xedges, yedges)
    )
    H = H.T
    # print(int(H.shape[1]/2))
    # West longitudeに直す
    # Ha = np.hstack((H[:, int(H.shape[1]/2):], H[:, 0:int(H.shape[1]/2)]))

    X, Y = np.meshgrid(xedges, yedges)
    fig, ax = plt.subplots(figsize=(8, 4))
    # ax.set_aspect(1)
    ax.set_title('1000 eV, 12$^\\circ$-bin', fontsize=10)
    ax.set_xlabel('West Longitude', fontsize=9)
    ax.set_ylabel('Colatitude', fontsize=9)
    ax.set_xticks(np.linspace(-180, 180, 5))
    ax.set_yticks(np.linspace(0, 180, 5))
    ax.set_xticklabels(['360$^\\circ$W', '270$^\\circ$W', '180$^\\circ$W',
                        '90$^\\circ$W', '0$^\\circ$'])
    ax.set_yticklabels(['0$^\\circ$', '45$^\\circ$', '90$^\\circ$',
                        '135$^\\circ$', '180$^\\circ$'])
    # ax.set_xlim([0,2*np.pi])
    # ax.set_ylim([-1.1,1.1])
    # ax.plot(np.pi,0, marker='o', color=color[5])
    ax.invert_yaxis()
    mappable0 = ax.pcolormesh(X, Y, H, cmap='magma',
                              vmin=0)
    pp = fig.colorbar(mappable0, orientation='vertical')
    # pp.set_clim(0, 100)
    pp.set_label('e$^-$ s$^{-1}$ area$^{-1}$', fontname='Arial', fontsize=10)
    fig.tight_layout()

    # stop = time.time()  # 時間計測終了
    # print('execution time: %.3f sec' % (stop - start))  # 計算時間表示

    plt.show()
    # plt.close()

    return 0


# %% main関数
def main():
    filepath0 = '/Users/shin/Documents/Research/Europa/Codes/test_eury_17_20211021_5.txt'
    # euryの値は読み込むデータに応じて変更
    a0 = np.loadtxt(filepath0)
    a0[:, 0] += R0   # 座標原点を木星に
    a0_rot = rot_dipole(a0, lam)
    eur_rot = rot_dipole2(np.array([eurx+R0, eury, eurz]), lam)
    relative = a0_rot-eur_rot
    # print((relative)/REU)
    a0_map = prefind(relative)

    print('count: {:>7d}'.format(a0[:, 0].size))
    mapplot(a0_map)

    return 0


# %%
if __name__ == '__main__':
    a = main()
