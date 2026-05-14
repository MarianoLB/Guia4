# -*- coding: utf-8 -*-
"""
Funciones para FEM
"""
import numpy as np

def k_elemental(MC,MN,gl,elemento='barra'):
    
    nxel = len(MC[0])
    x = np.zeros([len(MC), nxel])
    y = np.zeros([len(MC), nxel])
    L = np.zeros(len(MC))

    if elemento=="barra": #barra axial 1D
        kel = np.zeros([len(MC), nxel*gl, nxel*gl])   # una kel por elemento
        for e in range(len(MC)):
            for i in range(len(MC[e])):
                x[e,i] = MN[MC[e,i],0]
                y[e,i] = MN[MC[e,i],1]
                
                L[e]=np.sqrt((x[e,1]-x[e,0])**2 + (y[e,1]-y[e,0])**2 )  

                kel[e] = np.array([ [1,-1],
                                    [-1,1]  ])
                
                kel[e][np.abs(kel[e]) < 1e-10] = 0    
    
    elif elemento=="puente": #barras 2D con cosenos y senos
        #viga con dy y phi
        nxel = len(MC[0])
        ang = np.zeros(len(MC))
        L = np.zeros(len(MC))
        kel = np.zeros([len(MC), nxel*gl, nxel*gl])   # una kel por elemento

        for e in range(len(MC)):
            for i in range(len(MC[e])):
                x[e,i] = MN[MC[e,i],0]
                y[e,i] = MN[MC[e,i],1]
                
                ang[e] = np.arctan2((y[e,1]-y[e,0]), (x[e,1]-x[e,0]))
                L[e]=np.sqrt((x[e,1]-x[e,0])**2 + (y[e,1]-y[e,0])**2 )  
                c = np.cos(ang[e])
                s = np.sin(ang[e])
                kel[e] = np.array([ [ c**2,   c*s,  -c**2,  -c*s],[ c*s,   s**2,  -c*s,   -s**2],
                                   [-c**2, -c*s,   c**2,    c*s],  [-c*s,  -s**2,  c*s,    s**2]])

            kel[e][np.abs(kel[e]) < 1e-10] = 0
            
    elif elemento=="viga":
        kel=np.zeros([len(MC),nxel*gl,nxel*gl])
        
        for e in range(len(MC)):
            for i in range(len(MC[e])):
                x[e,i]=MN[MC[e,i],0]
                y[e,i]=MN[MC[e,i],1]
            
            L[e]=np.sqrt((x[e,1]-x[e,0])**2+(y[e,1]-y[e,0])**2)
            Le=L[e]
            
            kel[e]=np.array([[12,6*Le,-12,6*Le],
                             [6*Le,4*Le**2,-6*Le,2*Le**2],
                             [-12,-6*Le,12,-6*Le],
                             [6*Le,2*Le**2,-6*Le,4*Le**2]])
            
            kel[e][np.abs(kel[e])<1e-10]=0
    return kel,L
'''           
    elif elemento=="triangulo":
        nxel=len(MC[0])
        x=np.zeros([len(MC),nxel])
        y=np.zeros([len(MC),nxel])
        beta=np.zeros((len(MC),nxel))
        gamma=np.zeros((len(MC),nxel))
        A=np.zeros(len(MC))
        B=np.zeros([len(MC),3,gl*nxel])
        kel=np.zeros([len(MC),gl*nxel,gl*nxel])
    
        D=(E/(1-v**2))*np.array([[1,v,0],
                                     [v,1,0],
                                 [0,0,(1-v)/2]])
    
        for e in range(len(MC)):
            for i in range(nxel):
                x[e,i]=MN[MC[e,i],0]
                y[e,i]=MN[MC[e,i],1]
            
            beta[e,:]=[y[e,1]-y[e,2],y[e,2]-y[e,0],y[e,0]-y[e,1]]
            gamma[e,:]=[x[e,2]-x[e,1],x[e,0]-x[e,2],x[e,1]-x[e,0]]
        
            A[e]=0.5*np.linalg.det(np.array([[1,x[e,0],y[e,0]],[1,x[e,1],y[e,1]],[1,x[e,2],y[e,2]]]))
        
        B[e,:,:]=(1/(2*A[e]))*np.array([[beta[e,0],0,beta[e,1],0,beta[e,2],0],
                                      [0,gamma[e,0],0,gamma[e,1],0,gamma[e,2]],
                 [gamma[e,0],beta[e,0],gamma[e,1],beta[e,1],gamma[e,2],beta[e,2]]])
        
        kel[e,:,:]=t*abs(A[e])*np.transpose(B[e,:,:])@D@B[e,:,:]
        
        kel[e][np.abs(kel[e])<1e-10]=0
    
    L=A        
'''        
    
#-----------------------------------------------------------------------
def k_global(MC, MN, gl, kel):
    kglob = np.zeros([gl*len(MN), gl*len(MN)])

    for e in range(len(MC)):
        for i in range(len(MC[e])):
            rangoi = np.linspace(i*gl, (i+1)*gl-1, gl, dtype=int)
            rangoni = np.linspace(MC[e,i]*gl, (MC[e,i]+1)*gl-1, gl, dtype=int)
            for j in range(len(MC[e])):
                rangoj = np.linspace(j*gl, (j+1)*gl-1, gl, dtype=int)
                rangonj = np.linspace(MC[e,j]*gl, (MC[e,j]+1)*gl-1, gl, dtype=int)
                kglob[np.ix_(rangoni, rangonj)] = kglob[np.ix_(rangoni, rangonj)] + kel[e][np.ix_(rangoi, rangoj)]

    kglob[np.abs(kglob) < 1e-10] = 0
    return kglob

#-------------------------------------------
def resolver(MC,MN,gl,kel,F,s):
    
    K=k_global(MC,MN,gl,kel)
    
    s=np.array(s)
    todos=np.arange(gl*len(MN))
    r=np.setdiff1d(todos,s)
    
    u=np.zeros(len(K))
    
    krr=K[np.ix_(r,r)]
    krs=K[np.ix_(r,s)]
    
    b=F[r]-krs@u[s]
    Ku=np.linalg.solve(krr,b)
    
    u[r]=Ku
    
    Fint=K@u
    Fint[np.abs(Fint)<1e-10]=0
    
    R=Fint-F
    R[np.abs(R)<1e-10]=0
    
    U=u.reshape(len(MN),gl)
    Fnod=Fint.reshape(len(MN),gl)
    Reac=R.reshape(len(MN),gl)
    
    return K,u,U,Fint,Fnod,R,Reac,r

#_------------------------------------------
def esfuerzos(MC,MN,gl,u,D,B):
    
    nxel=len(MC[0])
    sig=np.zeros((len(MC),3))
    
    for e in range(len(MC)):
        dl=np.zeros(gl*nxel)
        
        for i in range(nxel):
            nodo=MC[e,i]
            dl[gl*i]=u[gl*nodo]
            dl[gl*i+1]=u[gl*nodo+1]
            
        sig[e,:]=D@B[e,:,:]@dl
    
    sig[np.abs(sig)<1e-10]=0
    
    return sig