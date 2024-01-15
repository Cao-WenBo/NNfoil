import numpy as np
from scipy.linalg import pinv
import matplotlib.pyplot as plt

# load airfoil

airfoil_up_file = "naca2412_up.dat"
data_up = np.loadtxt(airfoil_up_file, delimiter="\t")  # [5920 3]
airfoil_down_file = "naca2412_down.dat"
data_down = np.loadtxt(airfoil_down_file, delimiter="\t")  # [5760 3]

# CST hyperparameters
N = 5     
N1 = 0.5 
N2 = 1 


class CstCal:
    def __init__(self, degree, n_1, n_2):
        super(CstCal, self).__init__()
        self.degree = degree
        self.n_1 = n_1
        self.n_2 = n_2
        self.mx = None
        self.b = None

    def cal_mx(self, x_cor):
        num = x_cor.shape[0]
        self.mx = np.empty([num, self.degree+1])
        for index in range(self.degree+1):
            f = np.math.factorial(self.degree) / (np.math.factorial(index) * np.math.factorial(self.degree-index))
            self.mx[:, index] = f * np.power(x_cor, index+self.n_1) * np.power(1-x_cor, self.degree-index+self.n_2)
        return self.mx


    def cal_b(self, x_cor, y_cor):
        self.b = np.matmul(pinv(self.cal_mx(x_cor)), y_cor[:,np.newaxis])
        return self.b


    def cal_point(self, y_cor):
        points = np.matmul(self.mx, self.b)
        mse = np.mean(np.square(points-y_cor[:,np.newaxis]))
        return points, mse



cst_cal_up = CstCal(N, N1, N2)
B_up = cst_cal_up.cal_b(data_up[:,0], data_up[:,1])

Y_point_up, MSE_up = cst_cal_up.cal_point(data_up[:,1])


cst_cal_down = CstCal(N, N1, N2)
B_down = cst_cal_down.cal_b(data_down[:,0], data_down[:,1])

Y_point_down, MSE_down = cst_cal_down.cal_point(data_down[:,1])



data_X = np.concatenate((np.flip(data_down[:,0], axis=0), data_up[1:None,0]))
data_Y = np.concatenate((np.flip(Y_point_down, axis=0), Y_point_up[1:None,:]))
XY = np.concatenate((data_X[:,np.newaxis], data_Y), axis=1)

# save airfoil
tec_title = "TITTLE = 'Foil XY'\nVARIABLES = \"X\", \"Y\"\n"
tec_ZONE = "ZONE I=364, J=1, DATAPACKING=POINT"
file_airfoil = "airfoil_point.dat"
np.savetxt(file_airfoil, XY, fmt='%.9e', header=tec_title+tec_ZONE, comments="")


# save CST
CST = np.concatenate((B_up, B_down), axis=1)
file_CST = "airfoil_CST.dat"
np.savetxt(file_CST, CST, fmt='%.9e', delimiter=',')

plt.figure()
plt.plot(data_up[:,0],data_up[:,1],data_down[:,0],data_down[:,1],color='grey')
plt.plot(XY[:,0],XY[:,1], linestyle='dashed', linewidth=2, label='CST airfoil')