import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time as pytime
from numba import njit, jit
from IPython import display
import os
from scipy.optimize import minimize_scalar
import imageio

plt.style.use('dark_background')


############################### Shallow-water #############################################
@jit 
def R_hidraulico(A,Bw,i):
  return A[i]/(Bw[i]+2*A[i]/Bw[i])

@jit
def S_manning(Q,A,Bw,n,i):
  Rh=R_hidraulico(A,Bw,i)
  return (Q[i]*n)**2/(A[i]**2*Rh**(4.0/3.0))

@jit
def c_medio(g,A,Bw,i):
    return np.sqrt(abs(g*(A[i]+A[i+1])/(Bw[i]+Bw[i+1])))

@jit
def u_medio(A,Q,i):
    u=Q/A
    return(u[i]*np.sqrt(abs(A[i]))+u[i+1]*np.sqrt(abs(A[i+1])))/(np.sqrt(abs(A[i]))+np.sqrt(abs(A[i+1])))

@jit
def lambda1(g,A,Q,Bw,i):
    return u_medio(A,Q,i) - c_medio(g,A,Bw,i)

@jit
def lambda2(g,A,Q,Bw,i):
    return u_medio(A,Q,i) + c_medio(g,A,Bw,i)

@jit
def avec_1(g,A,Q,Bw,i):
    return np.array([1.0,lambda1(g,A,Q,Bw,i)])

@jit
def avec_2(g,A,Q,Bw,i):
    return np.array([1.0,lambda2(g,A,Q,Bw,i)])

@jit
def alpha1(g,A,Q,Bw,i):
    return(lambda2(g,A,Q,Bw,i)*(A[i+1]-A[i])-(Q[i+1]-Q[i]))/(2*c_medio(g,A,Bw,i))

@jit
def alpha2(g,A,Q,Bw,i):
    return(-lambda1(g,A,Q,Bw,i)*(A[i+1]-A[i])+(Q[i+1]-Q[i]))/(2*c_medio(g,A,Bw,i))

@jit
def beta1(g,n,Delta_x,A,Q,Bw,slope,i):
    c_ = c_medio(g,A,Bw,i)
    A_ = 0.5*(A[i]+A[i+1])
    B_ = 0.5*(Bw[i]+Bw[i+1])
    S0_ = -(slope[i+1]-slope[i])/Delta_x
    Sf_ = 0.5*(S_manning(Q,A,Bw,n,i)+S_manning(Q,A,Bw,n,i+1))                                           #Cambiar si se añade manning
    dA = A[i+1]-A[i]
    dQ = Q[i+1]-Q[i]
    dh = A[i+1]/Bw[i+1]-A[i]/Bw[i]                                                                                               
    return -(g*A_*((S0_-Sf_)*Delta_x-dh+dA/B_))/(2*c_)

@jit
def beta2(g,n,Delta_x,A,Q,Bw,slope,i):
    return -beta1(g,n,Delta_x,A,Q,Bw,slope,i)

@jit
def gamma1(g,n,Delta_x,A,Q,Bw,slope,i):
    return alpha1(g,A,Q,Bw,i)-beta1(g,n,Delta_x,A,Q,Bw,slope,i)/lambda1(g,A,Q,Bw,i) 

@jit
def gamma2(g,n,Delta_x,A,Q,Bw,slope,i):
    return alpha2(g,A,Q,Bw,i)-beta2(g,n,Delta_x,A,Q,Bw,slope,i)/lambda2(g,A,Q,Bw,i) 

@jit
def calcula_dt(g,A,Q,Bw,nx_cell,CFL,Delta_x):
    lambdas=[]
    for i in range(nx_cell-1):
      if A[i]!=0 or A[i+1]!=0:
        lambdas.append(abs(lambda1(g,A,Q,Bw,i)))
        lambdas.append(abs(lambda2(g,A,Q,Bw,i)))
    res = CFL*Delta_x/max(lambdas)
    
    return res
  
@jit
def flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i):
  ans=0.0
  lambda1_m = 0.5*(lambda1(g,A,Q,Bw,i)-abs(lambda1(g,A,Q,Bw,i)))
  lambda2_m = 0.5*(lambda2(g,A,Q,Bw,i)-abs(lambda2(g,A,Q,Bw,i)))
  
  if lambda1_m!=0:
    ans = ans + lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)

  if lambda2_m!=0:
    ans = ans + lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)
  return Q[i] + ans 

@jit
def lista_flujos(g:float,n:float,
                  A:np.ndarray,Q:np.ndarray,S:np.ndarray,
                  Bw:np.ndarray,slope:np.ndarray,
                  nx_cell,Delta_x):
                  
  flujos_right=np.zeros((nx_cell,2))
  flujos_left=np.zeros((nx_cell,2))
  flujos_num=np.zeros(nx_cell)#Q.copy()
  for i in range(nx_cell):
    lambda1_p = 0.5*(lambda1(g,A,Q,Bw,i-1)+abs(lambda1(g,A,Q,Bw,i-1)))
    lambda2_p = 0.5*(lambda2(g,A,Q,Bw,i-1)+abs(lambda2(g,A,Q,Bw,i-1)))
    lambda1_m = 0.5*(lambda1(g,A,Q,Bw,i)-abs(lambda1(g,A,Q,Bw,i)))
    lambda2_m = 0.5*(lambda2(g,A,Q,Bw,i)-abs(lambda2(g,A,Q,Bw,i)))

    if i==nx_cell-1:
      flujos_left[i,0] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[0]
      flujos_left[i,1] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[1]
      
      A_prima=np.append(A,A[-1])
      Q_prima=np.append(Q,Q[-1])
      Bw_prima=np.append(Bw,Bw[-1])
      slope_prima=np.append(slope,slope[-1])
      q_right=flujo_numerico(g,n,Delta_x,A_prima,Q_prima,Bw_prima,slope_prima,i)
      #q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
      flujos_num[i] += q_right*S[i] 
      #flujos_num[i] += q_left*S[i-1] if q_left>0 else q_left*S[i]

    if i==0:
      flujos_right[i,0] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[0]
      flujos_right[i,1] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[1]
      q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i)
      flujos_num[i] += q_right*S[i] if q_right>0 else q_right*S[i+1]

    if i>0 and i<(nx_cell-1):
      flujos_left[i,0] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[0]
      flujos_left[i,1] = lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[1]
      
      flujos_right[i,0] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[0]
      flujos_right[i,1] = lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[1]

      q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i)
      #q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
      flujos_num[i] += q_right*S[i] if q_right>0 else q_right*S[i+1]
      #flujos_num[i] += q_left*S[i-1] if q_left>0 else q_left*S[i]

  return flujos_left,flujos_right,flujos_num

@jit
def update(g:float,n:float,A:np.ndarray,Q:np.ndarray,S:np.ndarray,
            Bw:np.ndarray,slope:np.ndarray,
            nx_cell,CFL,Delta_x,mode,
            up_contA:np.ndarray,up_contQ:np.ndarray,up_contS:np.ndarray,down_contA:np.ndarray,down_contQ:np.ndarray,
            t,t_f):

  Delta_t = calcula_dt(g,A,Q,Bw,nx_cell,CFL,Delta_x)
  if t+Delta_t>t_f:
    Delta_t=t_f-t

  A_pres=np.zeros(nx_cell)
  Q_pres=np.zeros(nx_cell)
  S_pres=np.zeros(nx_cell)
  flujos_right=np.zeros((nx_cell,2))
  flujos_left=np.zeros((nx_cell,2))
  flujos_num=np.zeros(nx_cell)
  coef = (Delta_t/Delta_x)

  flujos_left,flujos_right,flujos_num = lista_flujos(g,n,A,Q,S,Bw,slope,nx_cell,Delta_x)



  for i in range(nx_cell):

    A_pres[i] = A[i]-coef*(flujos_left[i,0]+flujos_right[i,0])
    Q_pres[i] = Q[i]-coef*(flujos_left[i,1]+flujos_right[i,1])

    #soluto
    S_pres[i] = (A[i]*S[i]-coef*(flujos_num[i]-flujos_num[i-1]))/A_pres[i]

  if mode=='free':
    pass

  if mode=='supercritical':#Supercritico
    A_pres[0]=up_contA
    Q_pres[0]=up_contQ
    
  if mode=='subcritical':#NO EXISTE->Subcritico
    Q_pres[0]=up_contQ
    A_pres[-1]=down_contA

  if mode=='close':
    Q_pres[0]=0.0
    Q_pres[-1]=0.0

  S_pres[0]=up_contS

  return A_pres,Q_pres,S_pres,Delta_t
  
#@jit
def update2(g:float,n:float,A:np.ndarray,Q:np.ndarray,S:np.ndarray,Bw:np.ndarray,slope:np.ndarray,nx_cell,CFL,Delta_x,mode,up_contA:np.ndarray,up_contQ:np.ndarray,up_contS:np.ndarray,down_contA:np.ndarray,down_contQ:np.ndarray):

    Delta_t = calcula_dt(g,A,Q,Bw,nx_cell,CFL,Delta_x)
    A_pres=np.zeros(nx_cell)
    Q_pres=np.zeros(nx_cell)
    S_pres=np.zeros(nx_cell)
    coef = (Delta_t/Delta_x)

    for i in range(1,nx_cell-1):
      
      lambda1_p = 0.5*(lambda1(g,A,Q,Bw,i-1)+abs(lambda1(g,A,Q,Bw,i-1)))
      lambda2_p = 0.5*(lambda2(g,A,Q,Bw,i-1)+abs(lambda2(g,A,Q,Bw,i-1)))
      lambda1_m = 0.5*(lambda1(g,A,Q,Bw,i)-abs(lambda1(g,A,Q,Bw,i)))
      lambda2_m = 0.5*(lambda2(g,A,Q,Bw,i)-abs(lambda2(g,A,Q,Bw,i)))

      if lambda1_p!=0:
        A_pres[i] = A_pres[i] + lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[0]
        Q_pres[i] = Q_pres[i] + lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_1(g,A,Q,Bw,i-1)[1]

      if lambda2_p!=0:
        A_pres[i] = A_pres[i] + lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[0]
        Q_pres[i] = Q_pres[i] + lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,i-1)*avec_2(g,A,Q,Bw,i-1)[1]

      if lambda1_m!=0:
        A_pres[i] = A_pres[i] + lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[0]
        Q_pres[i] = Q_pres[i] + lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,i)*avec_1(g,A,Q,Bw,i)[1]

      if lambda2_m!=0:
        A_pres[i] = A_pres[i] + lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[0]
        Q_pres[i] = Q_pres[i] + lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,i)*avec_2(g,A,Q,Bw,i)[1]

      A_pres[i]=-coef*A_pres[i]
      Q_pres[i]=-coef*Q_pres[i]

      A_pres[i]=A_pres[i]+A[i]
      Q_pres[i]=Q_pres[i]+Q[i]

      #soluto
      ans = A[i]*S[i]
      q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i)
      q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
      ans -= coef*q_right*S[i] if q_right>0 else coef*q_right*S[i+1]
      ans += coef*q_left*S[i-1] if q_left>0 else coef*q_left*S[i]

      S_pres[i] = ans/A_pres[i]

      # A_pres[i]=0 if np.isnan(ans[0]) else ans[0]
      # Q_pres[i]=0 if np.isnan(ans[1]) else ans[1]

    lambda1_m = 0.5*(lambda1(g,A,Q,Bw,i)-abs(lambda1(g,A,Q,Bw,0)))
    lambda2_m = 0.5*(lambda2(g,A,Q,Bw,i)-abs(lambda2(g,A,Q,Bw,0)))
    lambda1_p = 0.5*(lambda1(g,A,Q,Bw,nx_cell-2)+abs(lambda1(g,A,Q,Bw,nx_cell-2)))
    lambda2_p = 0.5*(lambda2(g,A,Q,Bw,nx_cell-2)+abs(lambda2(g,A,Q,Bw,nx_cell-2)))
    

    if mode=='free':
      A_pres[0]=A_pres[1]
      A_pres[-1]=A_pres[-2]
      Q_pres[0]=Q_pres[1]
      Q_pres[-1]=Q_pres[-2]
      # A_pres[-1] = A[-1] + coef*(lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_1(g,A,Q,Bw,-2)[0]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_2(g,A,Q,Bw,-2)[0])
      # Q_pres[-1] = Q[-1] + coef*(lambda1_p*gamma1(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_1(g,A,Q,Bw,-2)[1]+lambda2_p*gamma2(g,n,Delta_x,A,Q,Bw,slope,-2)*avec_2(g,A,Q,Bw,-2)[1])
      # A_pres[0] = A[0] + coef*(lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,0)*avec_1(g,A,Q,Bw,0)[0]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,0)*avec_2(g,A,Q,Bw,0)[0])
      # Q_pres[0] = Q[0] + coef*(lambda1_m*gamma1(g,n,Delta_x,A,Q,Bw,slope,0)*avec_1(g,A,Q,Bw,0)[1]+lambda2_m*gamma2(g,n,Delta_x,A,Q,Bw,slope,0)*avec_2(g,A,Q,Bw,0)[1])


    if mode=='supercritical':#Supercritico
      A_pres[0]=up_contA
      Q_pres[0]=up_contQ
      A_pres[-1]=A_pres[-2]
      Q_pres[-1]=Q_pres[-2]

    if mode=='subcritical':#NO EXISTE->Subcritico
      A_pres[0]=A_pres[1]
      Q_pres[0]=up_contQ
      A_pres[-1]=down_contA
      Q_pres[-1]=Q_pres[-2]

    if mode=='both':
      A_pres[0]=up_contA
      Q_pres[0]=up_contQ
      A_pres[-1]=down_contA
      Q_pres[-1]=down_contQ

    if mode=='close':
      A_pres[0]=A_pres[1]
      A_pres[-1]=A_pres[-2]
      Q_pres[0]=0.0
      Q_pres[-1]=0.0

    S_pres[0]=up_contS
    S_pres[-1]=S_pres[-2]
    # A_prima   
    # ans = A[-1]*S[-1]
    #   q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i)
    #   q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
    #   ans -= coef*q_right*S[i] if q_right>0 else coef*q_right*S[i+1]
    #   ans += coef*q_left*S[i-1] if q_left>0 else coef*q_left*S[i]

    #   S_pres[i] = ans/A_pres[i]

    return A_pres,Q_pres,S_pres
    
#PLOTS

def plot_perfil_soluto(A,Q,S,x,Base_width,slope_z,t,t_real,rangeA=[0,0],rangeQ=[0,0],dir='./images/soluto',mc=0):
  fig2, ax2 = plt.subplots()
  color='tab:blue'
  ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; t={t_real:.2f} s')
  ax2.set_xlabel(f'$x [m]$')
  ax2.set_ylabel('$h[m]$',color=color)
  #ax2.grid()
  ax2.plot(x,A/Base_width+slope_z,'-',color=color)
  ax2.plot(x,S,'.',ms=2.5,color='green',label='$\Phi(x)$')
  ax2.set_ylim(rangeA) 
  #ax2.set_xlim([0,length])
  ax2.tick_params(axis='y', labelcolor=color)
  if sum(slope_z)!=0:
    ax2.fill_between(x,slope_z,0,color='grey')
  
  ax22=ax2.twinx()
  color='tab:red'
  ax22.set_ylabel('$Q[m^3/s]$',color=color)
  #ax22.grid()
  ax22.plot(x,Q,'-',color=color)
  ax22.set_ylim(rangeQ) 
  #ax22.set_xlim([0,length])
  ax22.tick_params(axis='y', labelcolor=color)
  
  if mc>0:
    xM, QM, hM, hzM, bM =np.array(import_MC(mc))
    ax2.plot(xM[::5],hzM[::5],'.',ms=2,color='cyan',label='McDonald')

  fig2.tight_layout()
  ax2.legend()
  quality=250

  fig2.savefig(dir+f'{t}.jpg', transparent=False,dpi=quality)#, facecolor='w')
  
def gifiyer(path,nt_cell,*,freeze=0,paso=1,FPS=10,gif_soluto=''):
  if len(gif_soluto)<2:
    gif_soluto='soluto_animation_mc.gif'
  images_data2 = []

  for i in range(freeze,1):
      data2 = imageio.imread(path+f'{i}.jpg')
      images_data2.append(data2)
      
  for i in range(1,nt_cell-1):
      if i%paso==0: 
          data2 = imageio.imread(path+f'{i+1}.jpg')
          images_data2.append(data2)

  imageio.mimwrite(gif_soluto, images_data2, format= '.gif', fps = FPS)

# Para busqueda del estacionario
@jit
def estabilidad(old,new):
  resta= old-new
  return abs(sum(resta/old))

#Para el adjunto
#@jit
def soluto_forward_plot(g,n,k,A,Q,S_inicio,Bw,slope,nx_cell,nt_cell,Delta_x,Delta_t,up_contS,freq=0,*,x_ax,Arange,Qrange):
  S_old = S_inicio.copy()
  S_new = S_inicio.copy()

  S_cont= np.zeros(nt_cell)
  S_cont[0]=S_new[-1]
  for t in np.arange(nt_cell):
    
    coef = (Delta_t/Delta_x)
    S_new=np.zeros(nx_cell)
    flujos_num=np.zeros(nx_cell)
    coef = (Delta_t/Delta_x)

    null,null,flujos_num = lista_flujos(g,n,A,Q,S_old,Bw,slope,nx_cell,Delta_x)
    for x in range(nx_cell):
      S_new[x] = (A[x]*S_old[x]-coef*(flujos_num[x]-flujos_num[x-1]))/A[x]-k*S_old[x]

    S_new[0]=up_contS[t]
    #S_new[-1]=S_new[-2]    
    S_old=S_new.copy()
    S_cont[t]=S_new[-1]

    if freq!=0 and t%freq==0:
      plot_perfil_soluto2(A,Q,S_new,x_ax,Bw,slope,t+1,t*Delta_t,Arange,Qrange)
      plt.close('all')

  return S_new,S_cont,nt_cell

# @njit
# def soluto_forward(g,n,k,A,Q,S_inicio,Bw,slope,nx_cell,nt_cell,Delta_x,Delta_t,up_contS):
#   S_old = S_inicio.copy()
#   S_new = S_inicio.copy()

#   S_cont= np.zeros(nt_cell)
#   S_cont[0]=S_new[-1]
#   for t in np.arange(nt_cell):
    
#     coef = (Delta_t/Delta_x)
#     S_new=np.zeros(nx_cell)
#     flujos_num=np.zeros(nx_cell)
#     coef = (Delta_t/Delta_x)

#     null,null,flujos_num = lista_flujos(g,n,A,Q,S_old,Bw,slope,nx_cell,Delta_x)
#     for x in range(nx_cell):
#       S_new[x] = (A[x]*S_old[x]-coef*(flujos_num[x]-flujos_num[x-1]))/A[x]-k*S_old[x]

#     S_new[0]=up_contS[t]
#     #S_new[-1]=S_new[-2]    
#     S_old=S_new.copy()
#     S_cont[t]=S_new[-1]

#   return S_new,S_cont,nt_cell
@jit
def soluto_forward(g,n,k,A,Q,S_inicio,Bw,slope,nx_cell,nt_cell,Delta_x,Delta_t,up_contS):
  S_old = S_inicio.copy()
  S_new = S_inicio.copy()
  
  S_cont= np.zeros(nt_cell)
  #S_cont[0]=S_new[-1]
  flujos_num=np.zeros(nx_cell)

  for x in range(nx_cell):
    if x==nx_cell-1:
      
      A_prima=np.append(A,A[-1])
      Q_prima=np.append(Q,Q[-1])
      Bw_prima=np.append(Bw,Bw[-1])
      slope_prima=np.append(slope,slope[-1])
      q_right=flujo_numerico(g,n,Delta_x,A_prima,Q_prima,Bw_prima,slope_prima,x)
      #q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
      flujos_num[x] += q_right 
    if x==0:
      q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,x)
      flujos_num[x] = q_right
    if x>0 and x<(nx_cell-1):
      
      q_right=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,x)
      #q_left=flujo_numerico(g,n,Delta_x,A,Q,Bw,slope,i-1)
      flujos_num[x] += q_right

  coef = (Delta_t/Delta_x)

  for t in np.arange(nt_cell):

    for x in range(1,nx_cell):
      ans = S_old[x]*A[x]
      if x==nx_cell-1:
        ans -= coef*flujos_num[x]*S_old[x]
        ans += coef*flujos_num[x-1]*S_old[x-1] if flujos_num[x-1]>0 else coef*flujos_num[x-1]*S_old[x]

        #S_new[x] = ans/A[x]

      if x>0 and x<nx_cell-1:
        ans -= coef*flujos_num[x]*S_old[x] if flujos_num[x]>0 else coef*flujos_num[x]*S_old[x+1]
        ans += coef*flujos_num[x-1]*S_old[x-1] if flujos_num[x-1]>0 else coef*flujos_num[x-1]*S_old[x]

      S_new[x] = ans/A[x]-k*S_old[x]

    S_new[0]=up_contS[t]
        
    S_old=S_new.copy()
    S_cont[t]=S_new[-1]

  return S_new,S_cont

  pass

#Función objetivo
def import_river():
  directorio="./river/shallow.txt"
  x, A, Q, Slope, b = zip(*(map(float, line.split()) for line in open(directorio,'r')))
  return x, A, Q, Slope, b 

@jit
def objetivo(simulacion, _medidas):
  Delta = (simulacion- _medidas)**2
  return 1/(2*len(Delta))*np.sum(Delta)

#evolucion adjunta solo hasta xm

@jit
def evolucion_inversa(soluto: np.ndarray, _medidas:np.ndarray,nx,nt,Dx,Dt,Q,A,K):
  times = (range(nt-1))
  Delta_x= Dx
  sigma_prev=np.zeros(nx) 
  sigma_new=np.zeros(nx)
  sigma_contorno=np.zeros(nt)
  for tinv in times:
    t = nt-2-tinv
    Delta_t=Dt
    sigma_prev=sigma_new.copy()
    for x in range(nx-1):
      
      Q_=Q[x]
      A_=A[x]
      u_=Q_/A_
      Dsig = sigma_prev[x+1]-sigma_prev[x]
      #Dsig_minus = _fases_sig[x,t+1]-_fases_sig[x-1,t+1]

      sigma_new[x] = sigma_prev[x] + Delta_t / Delta_x * (u_* Dsig)  #+  uplus * Dphi_plus) ??? Cambia el sentido de u
      
      # if x==0:
      #   _fases_sig[x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[x+2,t+1]-2*_fases_sig[x+1,t+1]+_fases_sig[x,t+1])
      # else:
      #   _fases_sig[x,t] += E*Delta_t/(Delta_x**2) * (_fases_sig[x+1,t+1]-2*_fases_sig[x,t+1]+_fases_sig[x-1,t+1])

      if (x+1)==(nx-1):
        aux = -(soluto[t+1]-_medidas[t+1]) * Delta_t / A_
        #print(aux)
        sigma_new[x] =sigma_new[x] + aux
      
      sigma_new[x] = sigma_new[x] + sigma_prev[x] * K * Delta_t

      #if _fases_sig[x,t]<0: _fases_sig[x,t] *= 0
    sigma_contorno[t]=sigma_new[0]

  return sigma_new,sigma_contorno
  

#Evolucion del contorno OOOOJOO QUE ESTOY COMPARANDO MEDIDAS Y AUX
@jit
def nuevo_cont_eps_i(g,n,k,A,Q,Bw,slope,nx_cell,nt_cell,Delta_x,Dt_list,up_contS, previo,gradiente,medidas,eps):
    
    aux = previo + gradiente*eps
    S_inicio=aux.copy()
    
    _,contorno=soluto_forward(g,n,k,A,Q,S_inicio,Bw,slope,nx_cell,nt_cell,Delta_x,Dt_list,up_contS)
    return objetivo(contorno,medidas)


@jit
def nuevo_cont_eps_c(g,n,k,A,Q,Bw,slope,nx_cell,nt_cell,Delta_x,Dt_list,previo, S_ini,gradiente,medidas,eps):
    
    aux = previo + gradiente*eps
    up_contS=aux.copy()
    
    _,contorno=soluto_forward(g,n,k,A,Q,S_ini,Bw,slope,nx_cell,nt_cell,Delta_x,Dt_list,up_contS)
    return objetivo(contorno,medidas)



#McDOnald Cases

def import_MC(caso):
  directorio="/home/sahara-rebel/Desktop/TFG/SIMULACIONES/macdonaldTestcases1-6/"
  x, Q, h, hz,_,_, b,*_ = zip(*(map(float, line.split()) for line in open(directorio+ f'{caso}-00.sol','r')))
  return x, Q, h, hz, b 

def plot_with_mc(caso,rangeA=[0,0],rangeQ=[0,0],dir='./images/mcdonald/'):

  x, Q, h, hz, b =np.array(import_MC(caso))
  slope_z=hz-h
  print('length:',len(x))

  fig2, ax2 = plt.subplots()
  color='cyan'
  ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; estacionario McDonald')
  ax2.set_xlabel(f'$x [m]$')
  ax2.set_ylabel('$h[m]$',color=color)
  #ax2.grid()
  ax2.plot(x,h+slope_z,'-',color=color)

  ax2.set_ylim(rangeA) 
  #ax2.set_xlim([0,length])
  ax2.tick_params(axis='y', labelcolor=color)
  if sum(slope_z)!=0:
    ax2.fill_between(x,slope_z,0,color='grey')
  
  ax22=ax2.twinx()
  color='tab:red'
  ax22.set_ylabel('$Q[m^3/s]$',color=color)
  #ax22.grid()
  ax22.plot(x,Q,'-',color=color)
  ax22.set_ylim(rangeQ) 
  #ax22.set_xlim([0,length])
  ax22.tick_params(axis='y', labelcolor=color)
  

  fig2.tight_layout()

  quality=90

  fig2.savefig(dir+f'mcdonald{caso}.jpg', transparent=False,dpi=quality)#, facecolor='w')
  #plt.show()


def plot_perfil_soluto2(A,Q,S,x,Base_width,slope_z,t,t_real,rangeA=[0,0],rangeQ=[0,0],dir='./images/soluto/'):
  fig2, ax2 = plt.subplots()
  color='tab:blue'
  ax2.set_title(f'Perfil de $U=(A,Q,\Phi)$; t={t_real:.2f} s')
  ax2.set_xlabel(f'$x [m]$')
  ax2.set_ylabel('$h[m]$',color=color)
  #ax2.grid()
  ax2.plot(x,A/Base_width+slope_z,'-',color=color)
  ax2.plot(x,S+slope_z,'.',ms=2.5,color='green',label='$\Phi(x)$')
  ax2.set_ylim(rangeA) 
  #ax2.set_xlim([0,length])
  ax2.tick_params(axis='y', labelcolor=color)
  if sum(slope_z)!=0:
    ax2.fill_between(x,slope_z,0,color='grey')
  
  ax22=ax2.twinx()
  color='tab:red'
  ax22.set_ylabel('$Q[m^3/s]$',color=color)
  #ax22.grid()
  ax22.plot(x,Q,'-',color=color)
  ax22.set_ylim(rangeQ) 
  #ax22.set_xlim([0,length])
  ax22.tick_params(axis='y', labelcolor=color)
  

  fig2.tight_layout()
  ax2.legend()
  quality=150

  fig2.savefig(dir+f'{t}.jpg', transparent=False,dpi=quality)#, facecolor='w')
  