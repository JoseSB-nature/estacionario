from functions import *

#Definir las variables del canal

nx_cell=401
#nt_cell=500
CFL=0.8
mode='subcritical' #puede ser 'free','up','down','both'
L = 150.0           
k_r = 0.0001
E_diff = 0.0
manning = 0.03
gravedad = 9.8

Base_width = np.array([1.0 if i<nx_cell/L*50 else 1.0 for i in range(nx_cell)])
Slope_z = np.linspace(0.1,0.0,nx_cell)

Delta_x = L/nx_cell

time =400

tolerancia=1e-9

#VARIABLES ACTUALIZABLES
A_old=np.zeros(nx_cell)
A_new=np.zeros(nx_cell)

Q_old=np.zeros(nx_cell)
Q_new=np.zeros(nx_cell)

S_old=np.zeros(nx_cell)
S_new=np.zeros(nx_cell)

Dt=0


#AREA Y CAUDAL
A_inicio=np.array([10.0 if i<nx_cell/L*50 else 10.0 for i in range(nx_cell)])
Q_inicio = np.array([0]*nx_cell)

q_up=2.0
a_up=1.0*Base_width[0]

q_down=0.5
a_down=1.0*Base_width[0]

#SOLUTO
s_up=0.0
S_inicio=np.array([2.0 if i<nx_cell/L*70 and i>nx_cell/L*30 else 0.0 for i in range(nx_cell)])
xmed=20
sig=10
x_0 = (np.array(range(nx_cell)))* L/nx_cell
#S_inicio= 1*np.exp(-(x_0-xmed)**2/(sig**2))

#CONTORNOS
contorno_up=[a_up,q_up,s_up]
contorno_down=[a_down,q_down,0.0]

### ADJUNTOS ###
time_soluto=160

#Para plots
x_axis = np.array(range(nx_cell))*Delta_x
Range_A=[0,12]
Range_Q=[0.0,25.0]
freq=100
mac=0