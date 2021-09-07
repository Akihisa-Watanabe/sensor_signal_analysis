import numpy as np



class KalmanFilter: ###kf4init
    def __init__(self, dt): 
        self.dt = dt
        self.Q = np.diag([10.74E-2*self.dt*self.dt, 10.74E-2*self.dt*self.dt]) #システム雑音の共分散行列
        self.R = np.diag([100.74E-2*self.dt*self.dt, 100.74E-2*self.dt*self.dt]) #観測雑音の共分散行列
        self.x_hat = np.array([[0.0],[0.0]]) #初期状態
        #self.x_pre = np.array([[0.0],[0.0]]) 
        self.z_pre = np.array([[0.0],[0.0]]) 
        self.z = np.array([[0.0],[0.0]]) #初期状態
        self.P_hat = np.diag([10.0,10.0])  #初期状態
        

    def get_angle(self, acc):
        """
        加速度から角度を計算して，これを観測値として使う
        """
        obs_angle = np.array([
            [np.arctan(acc[1]/acc[2])], 
            [-np.arctan(acc[0]/np.sqrt(acc[1]**2+acc[2]**2))]
            ])

        return obs_angle

    def f(self, gyro, x):
        '''状態方程式
            引数：
                x : k-1時点の状態ベクトル
            返り値：
                status : k-1時点の状態ベクトルと状態方程式を用いて予測したk時点の状態ベクトル
        '''

        phi = x[0][0]
        theta = x[1][0]
        cos1, sin1, tan1 = np.cos(phi), np.sin(phi), np.tan(phi) #phi
        cos2, sin2, tan2 = np.cos(theta), np.sin(theta), np.tan(theta)  #theta

        status = np.array([
            [phi+np.deg2rad(self.dt*gyro[0]) + np.deg2rad(self.dt*gyro[1])*sin1*tan2 + np.deg2rad(self.dt*gyro[2])*cos1*tan2],
            [theta+np.deg2rad(self.dt*gyro[1])*cos1-np.deg2rad(self.dt*gyro[2])*sin1],
            ])

        return status

    def matF(self, gyro, x):
        #状態方程式のヤコビアン
        phi = x[0][0]
        theta = x[1][0]
        cos1, sin1, tan1 = np.cos(phi), np.sin(phi), np.tan(phi) #phi
        cos2, sin2, tan2 = np.cos(theta), np.sin(theta), np.tan(theta)  #theta
        F = np.array([
            [1+np.deg2rad(self.dt*gyro[1])*cos1*tan2 - self.dt*gyro[2]*sin1*tan2, np.deg2rad(self.dt*gyro[1])*sin1/(cos2**2) + np.deg2rad(self.dt*gyro[2])*cos1/(cos2**2)],
            [-np.deg2rad(self.dt*gyro[1])*sin1 + np.deg2rad(self.dt*gyro[2])*cos1,  1.0]
            ])
        
        return F

    def matH(self): ###kf4funcs
        #観測モデルのヤコビアン
        return np.array([[1.0, 0.0], [0.0, 1.0]])


    def observation_update(self,acc):  
        if (abs(acc[0]) < 1e-5):
            acc[0] = 1e-5
        elif (abs(acc[1]) < 1e-5):
            acc[1] = 1e-5
        elif (abs(acc[2]) < 1e-5): 
            acc[2] = 1e-5 #値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる

        H = self.matH()
        self.z = self.get_angle(acc)
        self.z_pre = H.dot(self.x_pre) #一段先予測値
        K = self.P_pre.dot(H.T).dot(np.linalg.inv(self.R + H.dot(self.P_pre).dot(H.T)))
        self.x_hat = self.x_pre + K.dot(self.z - self.z_pre)
        self.P_hat = (np.eye(2) - K.dot(H)).dot(self.P_pre)
        

        
    def prediction(self, gyro): #追加
        if (abs(gyro[0]) < 1e-5):
            gyro[0] = 1e-5
        elif (abs(gyro[1]) < 1e-5):
            gyro[1] = 1e-5
        elif (abs(gyro[2]) < 1e-5): 
            gyro[2] = 1e-5 #値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる

        F = self.matF(gyro, self.x_hat)
        self.P_pre = F.dot(self.P_hat).dot(F.T) + self.Q
        self.x_pre = self.f(gyro, self.x_hat)


