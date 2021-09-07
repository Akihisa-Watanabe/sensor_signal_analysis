import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from  DOF6_EKF import KalmanFilter

FILE_NAME = "/Data/imu_data.csv"
dt = 0.01 #サンプリング周期

def read_dat(filename):
    dat = np.loadtxt(filename, delimiter=',',dtype='float')
    N = len(dat)
    acc = dat[:,1:4]
    gyro = dat[:,4:7]
    return acc, gyro, N


acc, gyro, N = read_dat(FILE_NAME)
ekf = KalmanFilter(dt)
x = np.zeros((N,2,1))
y = np.zeros((N,2,1))
P = np.zeros((N,2,2))
for i in range(N):
    ekf.prediction(gyro[i])
    ekf.observation_update(acc[i])
    x[i] = np.rad2deg(ekf.x_hat)
    y[i] = np.rad2deg(ekf.z)
    P[i] = ekf.P_hat





#Graph
time = np.arange(0, N*dt, dt) # 時間軸
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1) 
ax1.plot(time,y[:,0],label="Unfiltered")
ax1.plot(time,x[:,0],label="Filtered")
ax1.set_xlabel("time[s]")
ax1.set_ylabel("Roll[deg]")
ax1.legend()
 
ax2 = fig.add_subplot(2, 1,2) 
ax2.plot(time,y[:,1],label="Unfiltered")
ax2.plot(time,x[:,1],label="Filtered")
ax2.set_xlabel("time[s]")
ax2.set_ylabel("Pitch[deg]")
ax2.legend()
plt.show()
