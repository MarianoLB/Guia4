# -*- coding: utf-8 -*-
"""
Created on Fri May 15 16:42:49 2026

@author: LENOVO
"""

import gmsh
import numpy as np
from funciones_ef import k_elemental,k_global
from funciones_ef import resolver,esfuerzos,Triangulos

gmsh.initialize()

# Nombre del proyecto
gmsh.model.add("Ejercicio 1")

lc = 1 #Con lc=20 salen 4 elementos, mas chico=mas elementos

# Puntos
p1 = gmsh.model.geo.addPoint(0, 0, 0, lc)
p2 = gmsh.model.geo.addPoint(0, 10, 0, lc)
p3 = gmsh.model.geo.addPoint(20, 10, 0, lc)
p4 = gmsh.model.geo.addPoint(20, 0, 0, lc)

# Líneas
l1 = gmsh.model.geo.addLine(p1, p2)
l2 = gmsh.model.geo.addLine(p3, p2)
l3 = gmsh.model.geo.addLine(p3, p4)
l4 = gmsh.model.geo.addLine(p4, p1)

L = [l1, l2, l3, l4]

# Superficie
C1 = gmsh.model.geo.addCurveLoop([-l2, l3, l4, l1])
S = gmsh.model.geo.addPlaneSurface([C1])

gmsh.model.geo.synchronize()

# Borde izquierdo
Empotradox = gmsh.model.addPhysicalGroup(1, [l1])
gmsh.model.setPhysicalName(1, Empotradox, "Empotradox")

Empotradoy = gmsh.model.addPhysicalGroup(1, [l1])
gmsh.model.setPhysicalName(1, Empotradoy, "Empotradoy")

# Borde derecho traccionado
Traccionado = gmsh.model.addPhysicalGroup(1, [l3])
gmsh.model.setPhysicalName(1, Traccionado, "Traccionado")

# Superficie
Superficie = gmsh.model.addPhysicalGroup(2, [S])
gmsh.model.setPhysicalName(2, Superficie, "Superficie")

# Generar malla
#gmsh.option.setNumber("Mesh.Algorithm",32)  #para hacer solo 2 elementos

gmsh.model.mesh.generate(2)

# Guardar archivo
gmsh.write("t1.msh")

#Nodos
NodeInfo = gmsh.model.mesh.get_nodes_for_physical_group(2, Superficie)

Nodos = NodeInfo[0]
MN = NodeInfo[1].reshape(Nodos.shape[0], 3)
MNcontag = np.column_stack((Nodos, MN))

Info=gmsh.model.mesh.get_nodes_for_physical_group(1,Traccionado)
NodosT = Info[0]
MNTraccionado = Info[1].reshape(NodosT.shape[0], 3)
MNTraccionadocontag=np.column_stack((NodosT, MNTraccionado)) #Los tengo tageadoss

#Empotrado X
InfoX=gmsh.model.mesh.get_nodes_for_physical_group(1,Empotradox)
NodosEX = InfoX[0]
MNEX = InfoX[1].reshape(NodosEX.shape[0], 3)
MNEXcontag=np.column_stack((NodosEX, MNEX)) #Los tengo tageadoss

#Empotrado Y mm tal vez no me sirva mucho esto ja
InfoY=gmsh.model.mesh.get_nodes_for_physical_group(1,Empotradoy)
NodosEY = InfoY[0]
MNEY = InfoY[1].reshape(NodosEX.shape[0], 3)


#Elementos
Elements = gmsh.model.mesh.get_elements_by_type(2)

Elementos = Elements[0]
MC = Elements[1].reshape(Elementos.shape[0], 3) -1



#-----DATOS------------------
E=30e6 #psi
v=0.3
T=1000 #psi 
h=10
L=20
gl=2
t=1

nxel=len(MC[0])

kel,B,A,D=Triangulos(MC,MN,gl,E,v,t)

#---------------Distribucion de fuerzas------------------------------------------------
#Uso los MNraccionados, los ordenos en y asi saco las diferencias.
MNTord=MNTraccionadocontag[np.argsort(MNTraccionadocontag[:, 2])[::-1]]
long=np.zeros((len(MNTord[:,2]))-1)

F=np.zeros(gl * len(MN))

for i in range(len(MNTord[:,2]) - 1):

    long[i]=abs(MNTord[i+1, 2]- MNTord[i, 2]) #saco la long entre nodo osea de cada el
    n1=int(MNTord[i, 0]-1) #me fijo el n1 y n2 del elemento segun python
    n2=int(MNTord[i+1, 0]-1)

    Ft= T*t*long[i]
#Estos es para fuerzas en X si quiero en y gl*n+1
    F[gl*n1]+= Ft/2 #hago gl*n pq es para Fx
    F[gl*n2]+= Ft/2

#Vinculos, uso el empotrado
sx=(MNEXcontag[:, 0].astype(int)-1)*gl
sy=(MNEXcontag[:, 0].astype(int)-1)*gl+1
s = np.unique(np.concatenate((sx, sy))).astype(int)


K,u,U,Fint,Fnod,R,Reac,r=resolver(MC,MN,gl,kel,F,s)

sig=esfuerzos(MC,MN,gl,u,D,B)


# mostrar---------------------
# Desplazamientos como vectores
Desp_3d = np.column_stack((U[:,0], U[:,1], np.zeros(len(U))))

desp_view = gmsh.view.add("Desplazamientos")

gmsh.view.addModelData(desp_view,0,gmsh.model.getCurrent(),"NodeData",Nodos.astype(int),Desp_3d,time=0,numComponents=3)
# Tensiones de von Misess
sigma_x = sig[:, 0]
sigma_y = sig[:, 1]
tau_xy  = sig[:, 2]

von_mises = np.sqrt(sigma_x**2 - sigma_x*sigma_y + sigma_y**2 + 3*tau_xy**2)

stress_view = gmsh.view.add("Tensiones von Mises")

gmsh.view.addModelData(stress_view,0,gmsh.model.getCurrent(),"ElementData",Elementos.astype(int).tolist(),von_mises.reshape(-1, 1).tolist(),time=0,numComponents=1)

gmsh.fltk.run()

gmsh.finalize()