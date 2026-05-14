# -*- coding: utf-8 -*-
"""
Ejercicio 1
"""

import numpy as np
from funciones_ef import k_elemental,k_global
from funciones_ef import resolver,esfuerzos

#DATOS
E=30e6 #psi
v=0.3
T=5000 #psi 
h=10
L=20
gl=2
t=1

MC=np.array([[0,2,1],[0,3,2]])
MN=np.array([[0,0],[0,h],[L,h],[L,0]])
nxel=len(MC[0])


def Triangulos(MC,MN,gl,E,v,t=1):
    
    nxel=len(MC[0])
    x=np.zeros([len(MC),nxel])
    y=np.zeros([len(MC),nxel])
    B=np.zeros([len(MC),3,gl*nxel])
    beta=np.zeros((len(MC),nxel))
    gamma=np.zeros((len(MC),nxel))
    A=np.zeros(len(MC))
    kel=np.zeros((len(MC),gl*nxel,gl*nxel))
    
    D=(E/(1-v**2))*np.array([[1,v,0],
                             [v,1,0],
                             [0,0,(1-v)/2]])
    
    for e in range(len(MC)):
        for i in range(nxel):
            x[e,i]=MN[MC[e,i],0]
            y[e,i]=MN[MC[e,i],1]
            
        beta[e,:]=[y[e,1]-y[e,2],y[e,2]-y[e,0],y[e,0]-y[e,1]]
        gamma[e,:]=[x[e,2]-x[e,1],x[e,0]-x[e,2],x[e,1]-x[e,0]]
        
        A[e]=0.5*np.linalg.det(np.array([[1,x[e,0],y[e,0]],
                                         [1,x[e,1],y[e,1]],
                                         [1,x[e,2],y[e,2]]]))
        
        B[e,:,:]=(1/(2*A[e]))*np.array([
        [beta[e,0],0,beta[e,1],0,beta[e,2],0],
        [0,gamma[e,0],0,gamma[e,1],0,gamma[e,2]],
        [gamma[e,0],beta[e,0],gamma[e,1],beta[e,1],gamma[e,2],beta[e,2]]])
        
        kel[e,:,:]=t*abs(A[e])*np.transpose(B[e,:,:])@D@B[e,:,:]
        
        kel[e][np.abs(kel[e])<1e-10]=0
        B[e][np.abs(B[e])<1e-10]=0
        
    return kel,B,A,D

kel,B,A,D=Triangulos(MC,MN,gl,E,v,t)

#K=k_global(MC, MN, gl, kel)

#print((0.91/375000)*K)
#print(2*A[1]*B) #compruebro lo de la teoria.

#vinculos
s=[0,1,2,3]# d

F=np.zeros(gl*len(MN))
F[4]=T
F[6]=T

K,u,U,Fint,Fnod,R,Reac,r=resolver(MC,MN,gl,kel,F,s)

sig=esfuerzos(MC,MN,gl,u,D,B)

#print("K=")
#print(K)
#print("u=",u)
print("U nodal [dx,dy]=")
print(U)
print("F aplicada=",F)
print("Fint=",Fint)


# pal otro 4 triangulo
MC=np.array([[0,4,1],[4,2,1],[0,3,4],[4,3,2]])
MN=np.array([[0,0],[0,h],[20,h],[20,0],[L/2,h/2]])

kel2,B,A,D=Triangulos(MC,MN,gl,E,v,t)

#K=k_global(MC, MN, gl, kel2)

#vinculos
s=[0,1,2,3]# d

F=np.zeros(gl*len(MN))
F[4]=T
F[6]=T
K,u,U,Fint,Fnod,R,Reac,r=resolver(MC,MN,gl,kel2,F,s)

sig=esfuerzos(MC,MN,gl,u,D,B)

print("U nodal [dx,dy]=")
print(U)
print("F aplicada=",F)
print("Fint=",Fint)

