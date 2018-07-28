# -*- coding: utf-8 -*-
"""
Name:
    simulator.py
Purpose:
    白色干渉のシミュレーター
Specification:
    
Environment:
    Python 3.5.1
    
"""

import pandas as pd
from pylab import *
from scipy import signal


class Light(object):
    """
    パラメータ
    ------------
    wl_c : (val) 中心波長[um]
    wl_bw : (val) 波長のバンド幅[um]
    wl_step : (val) 波長のステップ幅[um]

    属性
    ------------
    scale_ : (array) 走査鏡の変位[um]
    fringe_ : (array) 干渉縞[um]
    envelope_ : (array) 包絡線[um]
    ep_ : (val) 包絡線のピーク位置[index]
    fp_ : (val) 干渉縞のピーク位置[index]

    """

    def __init__(self, wl_c, wl_bw, wl_step=1 / 1000):
        self.wl_c = wl_c
        self.wl_bw = wl_bw
        self.wl_step = wl_step
        self.wl_list_ = np.arange(wl_c - wl_bw / 2 * 2, (wl_c + wl_step) + wl_bw / 2 * 2, wl_step)  # 波長のリスト
        self.scale_ = None
        self.fringe_ = None
        self.envelope_ = None

    @staticmethod
    def search_neighbourhood(point, points, position='n'):
        """ある点(point)から最も近い点群(points)中の点の番号を返す"""
        if position == 'n':  # 最も近い点
            list = []
            for i in range(len(points)):
                l = abs(points[i] - point)
                list.append(l)
            return np.argmin(list)
        elif position == 'r':  # 最も近い右側の点
            if points[np.searchsorted(points, point)] >= point:
                return np.searchsorted(points, point)
            else:
                return np.searchsorted(points, point) + 1
        elif position == 'l':  # 最も近い左側の点
            if points[np.searchsorted(points, point)] <= point:
                return np.searchsorted(points, point)
            else:
                return np.searchsorted(points, point) - 1
        else:
            print('error')
            sys.exit()

    @staticmethod
    def ref_index_air(wl):
        n = -8e-10 * wl + 1.0003
        return n

    @staticmethod
    def ref_index_BK7(wl):
        B1 = 1.03961212E+00
        B2 = 2.31792344E-01
        B3 = 1.01046945E+00
        C1 = 6.00069867E-03
        C2 = 2.00179144E-02
        C3 = 1.03560653E+02
        n = np.sqrt(
            1 + B1 * (wl * wl) / (wl * wl - C1) + B2 * (wl * wl) / (wl * wl - C2) + B3 * (wl * wl) / (wl * wl - C3))
        return n

    @staticmethod
    def phase_shift(wl, material):
        params = {}
        params['Ag'] = (1.2104, -1.3392, 6.8276, 0.1761)
        params['Fe'] = (0.5294, -2.7947, 2.7647, 1.3724)
        params['Al'] = (1.3394, -0.6279, 11.297, -1.5539)
        params['Au'] = (0.6118, -0.3893, 6.4455, -0.1919)

        param = params[material]
        n = param[0] * wl + param[1]
        k = param[2] * wl + param[3]
        phi = np.arctan(-2 * k / (n * n + k * k - 1))
        return phi

    def I_gauss(self, wl):
        sigama2 = (self.wl_bw ** 2) / (8 * np.log(2))
        f = np.exp(-((wl - self.wl_c) ** 2) / (2 * sigama2)) / (np.power(2 * np.pi * sigama2, 0.5))
        return f

    def make_scale(self, scan_len, scan_step):
        self.scale_ = np.arange(-scan_len / 2, scan_len / 2 + scan_step, scan_step)

    def make_fringe(self, l_ref=3000 * 1000, l_bs=0, offset=0, material='BK7'):
        """スケールと干渉縞を作成"""
        fringe_list = []
        for wl in self.wl_list_:
            """あるwlでの干渉縞を作成"""
            print("making fringe")

            k_i = 2 * np.pi / wl
            intensity = self.I_gauss(wl)

            phi_x = k_i * self.scale_ * 2
            if material == 'BK7':
                phi_r = np.pi
            else:
                phi_r = self.phase_shift(wl, material)  # 反射での位相シフト(ガラス以外)
            phi_bs = k_i * (self.ref_index_BK7(wl) - self.ref_index_BK7(wl_c)) * l_bs * 2
            phi_offset = k_i * offset * 2
            phi = list(map(lambda x: x - phi_r - phi_bs - phi_offset + np.pi, phi_x))
            fringe = intensity * np.cos(phi)
            fringe_list.append(fringe)

        print("done")
        fringes = np.array(fringe_list)
        fringe_total = np.sum(fringes, axis=0)  # それぞれの波長での干渉縞を重ね合わせ
        self.fringe_ = fringe_total / max(fringe_total)
        self.envelope_ = abs(signal.hilbert(self.fringe_))

    def peak_detect(self):
        """EPとFPを検出"""
        self.ep_ = argmax(self.envelope_)
        fps = signal.argrelmax(self.fringe_)[0]  # 干渉縞の極大値のリスト
        self.fp_ = fps[self.search_neighbourhood(argmax(self.envelope_), fps)]

    def down_sample(self, step):
        self.scale_ = self.scale_[::step]
        self.fringe_ = self.fringe_[::step]
        self.envelope_ = abs(signal.hilbert(self.fringe_))
        self.peak_detect()

    def plot(self, ax):
        """描画"""
        ax.plot(self.scale_, self.fringe_)
        ax.plot(self.scale_, self.envelope_)
        ax.plot(self.scale_[self.fp_], self.fringe_[self.fp_], "bo")
        ax.plot(self.scale_[self.ep_], self.envelope_[self.ep_], "go")
        ax.grid(which='major', color='black', linestyle='-')
        ax.legend(["fringe", "envelope"])

    def print(self):
        """EPとFPの位置を出力"""
        print("ep: " + str(round(self.scale_[self.ep_], 3)) + "um",
              "fp: " + str(round(self.scale_[self.fp_], 3)) + "um")


if __name__ == '__main__':
    import copy

    def write_list(result, c_name, file='result.csv'):
        """結果をCSVファイルに書き込み"""
        df = pd.DataFrame(result)
        df.columns = c_name
        df.to_csv(file)


    wl_c = 1555 / 1000  # 中心波長[um]
    wl_bw = 50 / 1000  # バンド幅(FWHM)[um]
    scan_len = 80  # スキャン長さ[um]
    scan_step = 1 / 1000
    l_bs = 0  # BSの長さ[um]
    offset = 0

    # 基準干渉縞作成
    light = Light(wl_c, wl_bw, wl_step=1 / 100)
    light.make_scale(scan_len, scan_step)
    light.make_fringe(l_bs=l_bs, offset=offset, material='BK7')
    light.peak_detect()

    # 干渉縞複製・計算
    light2 = copy.deepcopy(light)
    light2.down_sample(100)

    # # 表示
    # fig1 = plt.figure()
    # ax1 = fig1.add_subplot(211)
    # ax2 = fig1.add_subplot(212)
    # light.plot(ax1)
    # light2.plot(ax2)

    plt.show()
    light.print()
    light2.print()

#    write_list([[x, y] for x, y in zip(light.scale_, light.fringe_)], ['position', 'intensity'])



    """L_bsを変更しながら計算"""
    """
    peaks = []
    for i in range(100):
        l_bs = i * 10
        light.make_fringe(l_bs=l_bs)
        light.peak_detect()
        peak = [round(light.scale_[light.ep_], 3), round(light.scale_[light.fp_], 3)]
        peaks.append(peak)
    write_list(peaks)
    """
