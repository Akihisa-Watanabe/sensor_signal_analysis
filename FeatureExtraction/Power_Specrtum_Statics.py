# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
import pandas as pd


Fs = 50  # サンプリング周波数
dt = 1/Fs #サンプリング周期
Fl = 6  # ローパス・フィルタ遮断周波数
Nf = 2  # ハイパス・フィルタ/ローパス・フィルタの次数


FILE_NAME = "Data/IMU_Example.csv"
CH =3

# フィルタの設計
bl, al = signal.butter(Nf, Fl, 'low', fs=Fs) #バターワースローパスフィルタ

def getFFT(frame):
    LEN = len(frame)
    han = signal.get_window("hann", LEN)
    enbw = LEN * np.sum(han**2) / np.sum(han)**2
    rfft_dat = rfft(frame * han)
    rfft_freq = rfftfreq(LEN, d=1.0/Fs)
    sp_rdat = np.abs(rfft_dat) ** 2 / (Fs * LEN * enbw)
    return sp_rdat, rfft_freq


def read_dat(filename, CH):
    dat = np.loadtxt(filename, delimiter=',')
    dat = dat[:,CH]
    return dat

def getMNP(fftdata,rfft_freq):
    """
    トータル・パワー
    """
    MNP = np.sum(fftdata)/len(fftdata)
    return  MNP

def getMNF(fftdata,rfft_freq):
    """
    平均周波数
    """
    MNF = np.sum(fftdata * rfft_freq) / np.sum(fftdata)
    return  MNF

def getMDF(fftdata,rfft_freq):
    """
    周波数中央値
    """
    ttp = np.sum(fftdata)
    area = np.cumsum(fftdata)
    area_half = ttp*0.5
    mdf_i = np.squeeze(np.where(area>=area_half))[0]
    mdf = rfft_freq[mdf_i]
    return mdf

def getPKF(fftdata,rfft_freq):
    "ピーク周波数"
    pkf_i = fftdata.argmax()
    pkf = rfft_freq[pkf_i]
    return pkf 



if __name__ == "__main__":
    # 生波形
    dat = read_dat(FILE_NAME,CH)
    time = np.arange(0, len(dat)*dt, dt) #時間軸
    plt.plot(time,dat)
    plt.show()
    dat = signal.lfilter(bl, al, dat) #Filtering
    
    sp_rdat,freq = getFFT(dat) 
    mnp = getMNP(sp_rdat,freq) 
    mnf = getMNF(sp_rdat, freq)
    mdf = getMDF(sp_rdat, freq) 
    pkf = getPKF(sp_rdat, freq) 

    print("statistic : ")
    print("MNP(Mean Power):{0}".format(mnp))
    print("MNF(Mean Frequency):{0} ".format(mnf))
    print("MDF(Median Frequency):{0}".format(mdf))
    print("PKF(Peak Frequency):{0} ".format(pkf))



