from functions import *
from variables import *

time_1=pytime.time()

#VACIAMOS DIRECTORIOS
path_1='./images/soluto'
path_3="./images/adjuntos"
path_2=path_1+'/animacion_soluto_t'

for filename in os.listdir(path_1):
    os.remove(path_1+'/'+filename)

#inicialización
A_old=A_inicio
Q_old=Q_inicio
S_old=S_inicio

#Cargamos macdonald
if mac>0:
    xM, QM, hM, hzM, bM =np.array(import_MC(mac))

    Base_width=bM
    Slope_z=hzM-hM
    a_up=hM[0]*bM[0]
    contorno_up[0]=a_up
    q_up=QM[0]
    contorno_up[1]=q_up


    a_down=hM[-1]*bM[-1]
    contorno_down[0]=a_down
    q_down=QM[-1]
    contorno_up[1]=q_down


plot_perfil_soluto(A_old,Q_old,S_old,
                    x_axis,Base_width,Slope_z,
                    0,0,
                    Range_A,Range_Q,path_2,mac)


time_2=pytime.time()

print(f"tiempo inicialización:{time_2-time_1:.2f}s")

#bucle de actualización
medidas_cont=[]
medidas_cont.append(S_inicio[-1])

#Esta=estabilidad(Q_old,Q_new)<tolerancia
t=1
real_t=0

while real_t<time:
    time_1=pytime.time()
    A_new,Q_new,S_new,D_t = update(gravedad,manning,
                                A_old,Q_old,S_old,
                                Base_width,Slope_z,
                                nx_cell,CFL,Delta_x,
                                mode,
                                contorno_up[0],contorno_up[1],contorno_up[2],
                                contorno_down[0],contorno_down[1],real_t,time)
    #Guardamos la malla
    real_t+=D_t
    if t%freq==0:
        plot_perfil_soluto(A_new,Q_new,S_new,
                        x_axis,Base_width,Slope_z,
                        t+1,real_t,
                        Range_A,Range_Q,path_2,mac)
    
        plt.close('all')

    #Esta=estabilidad(Q_old,Q_new)<tolerancia

    


    A_old=A_new
    Q_old=Q_new
    S_old=S_new

    time_2=pytime.time()
    if t==1:print(f"tiempo evolución:{time_2-time_1:.4f}s")
    medidas_cont.append(S_new[-1])

    

    t+=1

medidas_ini = S_new.copy()

plot_perfil_soluto(A_old,Q_old,S_old,
                    x_axis,Base_width,Slope_z,
                    t,real_t,
                    Range_A,Range_Q,path_2,mac)

#plt.show()

print(f"Tiempo transitorio({t}):",real_t," s")

plot_with_mc(3,Range_A,Range_Q)

#gif
time_1=pytime.time()
gifiyer(path_2,t,FPS=10,paso=freq)
time_2=pytime.time()
print(f"tiempo gif:{time_2-time_1:.2f}s")

###################################################Guardamos el estacionario################################################3
save_path="./river/"

open(save_path+'shallow.txt', 'w').writelines(list('\t'.join(map(str, med_set)) + '\n' for med_set in zip(x_axis,A_new,Q_new,Slope_z,Base_width)))