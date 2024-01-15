import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.io
from torch.optim import lr_scheduler
import torch.nn.utils.weight_norm as weight_norm
import time
from util import jacobian_trans, fwd_gradients
import pickle

init_seed = 0
np.random.seed(init_seed)
torch.manual_seed(init_seed)
torch.cuda.manual_seed(init_seed)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class Net(torch.nn.Module):
    def __init__(self, layer_dim, X, device):
        super().__init__()

        self.X_mean = X.mean(0, keepdim=True)
        self.X_std = X.std(0, keepdim=True)
        
        self.num_layers = len(layer_dim)
        temp = []
        for l in range(1, self.num_layers):
            temp.append(weight_norm(torch.nn.Linear(layer_dim[l-1], layer_dim[l]), dim = 0))
            torch.nn.init.normal_(temp[l-1].weight)
        self.layers = torch.nn.ModuleList(temp)
        
    def forward(self, x):
        x = ((x - self.X_mean) / self.X_std) # z-score norm
        for i in range(0, self.num_layers-1):
            x = self.layers[i](x)
            if i < self.num_layers-2:
                x = torch.tanh(x)
        return x
    
class NNfoil():
    
    def __init__(self, layers, device, xx, yy):

        self.layers = layers
        self.device = device
        self.wall = {}
        self.far = {}
        self.log = {'losses':[], 'losses_b':[], 'losses_f':[], 'losses_s':[], 'Cl':[], 'time':[]}
        
        # CST hyper-parameters
        self.degree = 5; self.n_1 = 0.5; self.n_2 = 1; 
        
        # flow condition
        self.Ma = torch.tensor(0.4); self.alpha = torch.tensor(4/180*torch.pi)
        
        self.Preprocess(xx, yy)
        self.model = Net(self.layers, self.XE, self.device).to(self.device)

        #Obtain airfoil by cst (True) or mesh (False) ? if True, use mesh deformation based on the read mesh.
        self.use_cst = False  
        #NACA0012
        b_up = [0.171787,0.155338,0.161996,0.137638,0.145718,0.143815]
        b_down = [-0.171787,-0.155338,-0.161996,-0.137638,-0.145718,-0.143815]
        
        self.b_up = torch.tensor(b_up).reshape(-1,1).to(self.device)
        self.b_down = torch.tensor(b_down).reshape(-1,1).to(self.device)
        
        self.Mesh()
        
    def Preprocess(self, xx, yy):
        #Initial Mesh
        self.xx0 = torch.tensor(xx.copy()).to(self.device).float()
        self.yy0 = torch.tensor(yy.copy()).to(self.device).float()
        self.X0 = torch.cat([self.xx0.reshape(-1,1),self.yy0.reshape(-1,1)],dim=1)
        
        #Initial airfoil
        self.wall['x0'] = self.xx0[:,0:1]
        self.wall['y0'] = self.yy0[:,0:1]
        self.mx = self.Cal_mx(self.wall['x0']+0.5).to(self.device)
        
        #computation domain
        self.M = xx.shape[0] #N_xi
        self.N = xx.shape[1] #N_eta
        
        self.xxi, self.eeta = torch.meshgrid(torch.arange(self.M), torch.arange(self.N))
        self.eeta, self.xxi= self.eeta.to(self.device).float(), self.xxi.to(self.device).float()
        self.XE = torch.cat([self.xxi.reshape(-1,1),self.eeta.reshape(-1,1)], dim=1)
        
        self.bond_left = torch.cat([self.xxi[0:1],self.eeta[0:1]], dim=0).T
        self.bond_right = torch.cat([self.xxi[-1:],self.eeta[-1:]], dim=0).T
        
        self.wall['XE'] = torch.cat([self.xxi[:,0:1],self.eeta[:,0:1]], dim=1)
        self.wall['XE_f'] = (self.wall['XE'][1:] + self.wall['XE'][:-1]) / 2
        self.far['XE'] = torch.cat([self.xxi[:,-1:],self.eeta[:,-1:]], dim=1)
        self.far['u'] = torch.cos(self.alpha)
        self.far['v'] = torch.sin(self.alpha)
        self.far['p'] = torch.tensor(1.0).to(self.device)/(1.4*self.Ma**2)
        self.far['rho'] = torch.tensor(1.0).to(self.device)
        
        #distance matrix for mesh deformation
        self.cdist = torch.sqrt((self.xx0-self.xx0[:,0:1])**2 + (self.yy0-self.yy0[:,0:1])**2)
        self.cdist = self.cdist / self.cdist[:,-1:]

    def Cal_mx(self, x_cor): 
        x_cor = x_cor.reshape(-1).cpu()
        num = x_cor.shape[0]
        mx = np.empty([num, self.degree+1])
        for index in range(self.degree+1):
            f = np.math.factorial(self.degree) / (np.math.factorial(index) * np.math.factorial(self.degree-index))
            mx[:, index] = f * np.power(x_cor, index+self.n_1) * np.power(1-x_cor, self.degree-index+self.n_2)
        mx = torch.tensor(mx).float()
        return mx
    
    def Mesh(self):
        if self.use_cst:
            #Mesh transformation requires that the upper and lower walls of the base mesh have the same number of grids and start from the leading edge or trailing edge.
            #It is recommended to use the provided grid as the base grid
            N_up = int(self.wall['y0'].shape[0]/2)
            self.wall['y'] = self.mx@self.b_up; self.wall['y'][N_up:] = self.mx[N_up:]@self.b_down
        else:
            self.wall['y'] = self.wall['y0']
        self.wall['X'] = torch.cat([self.wall['x0'], self.wall['y']], dim=1)
        self.wall['X0'] = torch.cat([self.wall['x0'], self.wall['y0']], dim=1)
        
        self.wall['X_f'] = (self.wall['X'][1:] + self.wall['X'][:-1]) / 2
        self.wall['dl_f'] = ((self.wall['X'][1:] - self.wall['X'][:-1]) ** 2).sum(1, keepdims=True) ** 0.5  # "_f" means face centre
        self.wall['tan_f'] = - (self.wall['X'][1:] - self.wall['X'][:-1]) / self.wall['dl_f']
        self.wall['nor_f'] = torch.cat([-self.wall['tan_f'][:,[1]], self.wall['tan_f'][:,[0]]], dim=1)

        dy = self.wall['y'] - self.wall['y0']
        self.yy = self.yy0 + (1 - self.cdist) * dy # 网格节点增量
        
        nodes = torch.arange(-3,4,1, device=self.device, dtype=torch.long)
        Jac, Jac_inv = jacobian_trans(self.xx0.double(), self.yy.double(), nodes, boundary=(1,1))
        self.J = torch.det(Jac).reshape(-1,1).float()
        self.Jac_inv = Jac_inv.reshape(-1,2,2).float()

    def Mseb(self):
    
        # far
        pred = self.model(self.far['XE'])
        u = pred[:,0:1]; v = pred[:,1:2]; p = pred[:,2:3]; rho = pred[:,3:4]
        mseb1 = ((p - self.far['p'])**2).mean() + ((u - self.far['u'])**2).mean() + \
            ((v - self.far['v'])**2).mean() + ((rho - self.far['rho'])**2).mean()
        
        # wall
        pred = self.model(self.wall['XE_f'])
        u_f = pred[:,0:1]; v_f = pred[:,1:2]; p_f = pred[:,2:3];
        vn_f = u_f*self.wall['nor_f'][:,0:1] + v_f*self.wall['nor_f'][:,1:2]
        mseb2 = (vn_f**2).mean()
        self.wall['Cp_f'] = (p_f-1/(1.4*self.Ma**2)) / (0.5)
        
        # interface
        X = self.bond_left
        U_left = self.model(X).reshape(-1,1)
        X = self.bond_right
        U_right = self.model(X).reshape(-1,1)
        mseb3 = F.mse_loss(U_left, U_right) 
        
        pred = self.model(self.wall['XE'])
        u = pred[:,0:1]; v = pred[:,1:2]; p = pred[:,2:3]; rho = pred[:,3:4]
        self.wall['u'] = u
        self.wall['v'] = v
        self.wall['rho'] = rho
        self.wall['p'] = p
        self.wall['Cp'] = (p-1/(1.4*self.Ma**2)) / (0.5)
        
        return mseb1 + mseb2 + mseb3

    def Msef(self):

        XE = self.XE[self.batch_ind]
        XE.requires_grad = True
        pred = self.model(XE)
        u = pred[:,0:1]; v = pred[:,1:2]; p = pred[:,2:3]; rho = pred[:,3:4]
        gamma = 1.4;
        u_xieta = fwd_gradients(u, XE)
        v_xieta = fwd_gradients(v, XE)
        p_xieta = fwd_gradients(p, XE)
        rho_xieta = fwd_gradients(rho, XE)
        
        u_xy = (self.Jac_inv[self.batch_ind]@u_xieta.reshape(-1,2,1)).squeeze()
        v_xy = (self.Jac_inv[self.batch_ind]@v_xieta.reshape(-1,2,1)).squeeze()
        p_xy = (self.Jac_inv[self.batch_ind]@p_xieta.reshape(-1,2,1)).squeeze()
        rho_xy = (self.Jac_inv[self.batch_ind]@rho_xieta.reshape(-1,2,1)).squeeze()
        
        u_x = u_xy[:,0:1]; u_y = u_xy[:,1:2]
        v_x = v_xy[:,0:1]; v_y = v_xy[:,1:2]
        p_x = p_xy[:,0:1]; p_y = p_xy[:,1:2]
        rho_x = rho_xy[:,0:1]; rho_y = rho_xy[:,1:2]
        
        a2 = gamma*p/rho
        e1 = rho*u_x + u*rho_x + rho*v_y + v*rho_y
        e2 = u*u_x + v*u_y + 1/rho*p_x
        e3 = u*v_x + v*v_y + 1/rho*p_y
        e4 = rho*a2*u_x +  rho*a2*v_y + u*p_x + v*p_y
        R = torch.stack([e1, e2, e3, e4], dim=2).view(u.shape[0], 4, 1)

        R_pre = R.squeeze()

        res_rho = R_pre[:,0:1]; res_u = R_pre[:,1:2]; res_v = R_pre[:,2:3]; res_p = R_pre[:,3:4];
        msef = ((res_rho**2 + res_u**2 + res_v**2 + res_p**2)*self.J[self.batch_ind]**2).sum() / (self.J[self.batch_ind]**2).sum()
        
        return msef

    def Calculate_Force(self):
        Force = self.wall['Cp_f']*self.wall['dl_f']*self.wall['nor_f']
        self.Cl = - (torch.cos(self.alpha)*Force[:,1:2] - torch.sin(self.alpha)*Force[:,0:1]).sum()
        self.Cd = - (torch.sin(self.alpha)*Force[:,1:2] + torch.cos(self.alpha)*Force[:,0:1]).sum()
        
    def Loss(self):
        loss_f = self.Msef()*2e4
        loss_b = self.Mseb()
        loss = loss_f + loss_b

        return loss, loss_b, loss_f
    
    def train(self, epoch):

        if len(self.log['time']) == 0:
            t1 = time.time()
        else:
            t1 = time.time() - self.log['time'][-1]

        for i in range(epoch):
            self.niter = i
            def closure():
                self.optimizer.zero_grad()
                self.loss, self.loss_b, self.loss_f = self.Loss()
                self.loss.backward()
                return self.loss

            # self.batch_ind = np.random.choice(self.XE.shape[0], 30000)
            self.batch_ind = np.random.choice(self.XE.shape[0], int(min(self.XE.shape[0]*0.8,20000)), replace=False)
            self.optimizer = torch.optim.LBFGS(self.model.parameters(), max_iter=1000, history_size=1000,
                                               tolerance_grad=1e-6, tolerance_change=1e-10)
            self.optimizer.step(closure)
            
            #exceptional handling
            if ((len(self.log['losses'])==0))and((self.loss != self.loss) or (self.loss_f < 1e-15)):
                print('Failed, Restart')
                self.model = Net(self.layers, self.XE, self.device).to(self.device)
                continue
            if (self.loss != self.loss)or(self.loss_f < 1e-15)or((len(self.log['losses'])>0)and(self.loss.item() > 4*self.log['losses'][-1])):
                print('Failed, Load previous model')
                self.model = torch.load('model_temp.pth')
                continue
            
            torch.save(self.model, 'model_temp.pth')
                
            t2 = time.time()
            self.log['losses'].append(self.loss.item())
            self.log['losses_f'].append(self.loss_f.item())
            self.log['losses_b'].append(self.loss_b.item())
            self.log['time'].append(t2-t1)
            
            print(f'{i}|{epoch} loss={self.loss.item()} PDE loss={self.loss_f.item()}')  

        
        
if __name__ == '__main__':
    
    t1 = time.time()
    torch.set_num_threads(1) 
    
    layers = [2, 128, 128, 128, 128, 128, 4]

    device = torch.device("cuda:0" if 1 else "cpu")

    xxyy = np.loadtxt('mesh/NACA0012_200_100.x', skiprows=1)
    # xxyy = np.loadtxt('mesh/NACA2412_200_100.x', skiprows=1)
    # xxyy = np.loadtxt('mesh/Airfoil1_200_100.x', skiprows=1)
    
    xx = xxyy[0:round(xxyy.shape[0]/2)].T
    yy = xxyy[round(xxyy.shape[0]/2):].T
    xx = np.r_[xx, xx[0:1]]
    yy = np.r_[yy, yy[0:1]]
 
    nn = NNfoil(layers, device, xx, yy)
    nn.train(20)
    t2 = time.time()
    print('wall time:', t2-t1)
    # %%plot
    from matplotlib import rcParams,colors
    my_font1 = {"family":"Times New Roman", "size":22, "style":"italic"}
    
    globalConfig = {
        "font.family":'serif',
        "font.size": 20,
        "font.weight": 'normal',
        "font.serif": ['Times New Roman'],
        "mathtext.fontset":'stix',
    }
    rcParams.update(globalConfig)
    
    plt.figure()
    plt.plot(nn.log['time'], np.log10(nn.log['losses']), label='loss')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_f']), label='loss_f')
    plt.plot(nn.log['time'], np.log10(nn.log['losses_b']), label='loss_b')

    plt.xlabel('wall time')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    
    # %%
    plt.figure()
    X = nn.xx0.cpu().detach()
    Y = nn.yy.cpu().detach()
    U = nn.model(nn.XE).cpu().detach().reshape(nn.M,nn.N,4)

    plt.contourf(X,Y,U[:,:,2])
    # plt.plot(X,Y,X.T,Y.T,color='grey', linewidth=0.5)
    plt.colorbar()
    plt.axis('equal')
    
    Cp = np.loadtxt('FVM_Cp/NACA0012_800_400_Cp.dat', skiprows=1)
    plt.figure()
    plt.plot(Cp[:,1:2],Cp[:,3:4], 'o', label='Ref')
    plt.plot(nn.wall['X'][:,0].cpu().detach(),nn.wall['Cp'][:,0].cpu().detach(), lw=2.0, label='NNfoil')
    plt.gca().invert_yaxis()
    plt.xlabel('$x$', fontdict=my_font1)
    plt.ylabel('$C_p$', fontdict=my_font1)
    plt.legend()
    plt.tight_layout()
    
    #%% write flow field to tecplot
    # with open('Flow.plt', 'w') as f:
    #     f.write('variables=x,y,u,v,p,rho\n')
    #     f.write('zone I= %d, J=%d\n'%(nn.N, nn.M))
    #     Data = torch.cat([X.reshape(-1,1), Y.reshape(-1,1), U.cpu().reshape(-1,4)], dim=1)
    #     for i in range(Data.shape[0]):
    #         f.write('%f, %f, %f, %f, %f, %f \n'%(Data[i,0],Data[i,1],Data[i,2],Data[i,3],Data[i,4],Data[i,5]))
    #     f.close()
