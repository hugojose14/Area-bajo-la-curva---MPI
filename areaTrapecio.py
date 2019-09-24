#Modulos utilizados
from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE
import numpy as np
from sys import argv
import time

#el numero de procesos en total
comm = MPI.COMM_WORLD
#El id para cada trabajo o tarea
rank = comm.Get_rank()
#el tamanio que tendra cada trabajo
size = comm.Get_size()

#Utilizando sys para pedir los argumentos por consola (terminal)  [a,b,tramos]
#Run programm: mpiexec -n 4 python areaTrapecio.py  [a,b,tramos] 
#Run programm: mpiexec -n 4 python areaTrapecio.py [1,100,division del numero de trapecios]
#mpiexec -n 4 python areaTrapecio.py 1 100 100000

#Aqui tendremos el tipo de datos que enviaremos por argumentos en la consola
a = float(argv[1])
b = float(argv[2])
#el numero de tramos que sera entero
tramos = int(argv[3])

#Funcion de prueba para la integral (cos (x) + x**3)
#Esta sera la funcion inicial para calcular la integral 
#luego dividiremos cada tramos entre el numero de procesos
function = lambda x : np.cos(x) + x**3

#Dividimos el numero de tramos (rango) entre el numero de procesos
def rango_integral(a,b,tramos):
    """ 
    integral = -(function(a) + function(b))/2.0
    endpoints = np.linspace(a, b, tramos+1)
    for x in (endpoints):
        integral+= function(x)
    integral *=(b-a)/tramos """
    
    endpoints = np.linspace(a, b, tramos+1)
    integral = sum(function(x) for x in endpoints)
    integral -= (function(a) + function(b))/2.0
    integral *= (b - a)/tramos
    return integral

#tramos = numero de trapezoides 
#h va a ser  el tamanio que vamos a dividir el trapezoide
h = (b-a)/tramos

#El numero de tramos que se calculara en cada proceso
local_tramos = tramos / size

#Calculo del intervalo para cada uno de los procesos
local_a = a + rank*local_tramos*h
local_b = local_a + local_tramos*h

#Inicializando las variables de mpi4py
#Iniciamos integral en un numpy zero (0) -> un objeto numpy
integral = np.zeros(1)
auxiliar = np.zeros(1)
dest = np.zeros(1)

#Calculo: El intervalo que tendra cada proceso 
#Integral (Vector)
integral[0] = rango_integral(local_a,local_b,local_tramos)

#Recibimos el resultado de todos los procesos y hacemos la suma para calcular el area total
#aqui haremos la comunicacion para cada uno de los procesos
if rank == 0:
    areaTotal = integral[0]
    for i in range(size-1):
        print("PE",rank,",",i,"area calculada en el tramo",areaTotal,"\n")
        comm.Recv(auxiliar, MPI.ANY_SOURCE)
        areaTotal += auxiliar[0]

#Aqui tendremos lo que envia cada proceso 
#envian la integral con cada uno de los parametros
#se coloca 0, debido a que Send recibe dos parametros y el por defecto seria 0 tambien
#dest seria el parametro 0 que recibe 0 en sus argumentos
else:
    comm.Send(integral,0)
    
#Mostrando los resultados (sumando cada  (tramo) dividido) y obteniendo el area total
#comunicacion == 0
if (comm.rank==0):
    print("Tramos",tramos,"Trapezoides", a, "a", b, "es", areaTotal)

    
    
    




    
    