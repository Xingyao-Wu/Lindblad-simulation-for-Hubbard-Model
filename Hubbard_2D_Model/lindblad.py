"""
Lindblad based method for ground state preparation

Lin Lin
Last revision: 3/28/2023
"""

import numpy as np # generic math functions
import scipy.sparse
import scipy.linalg as la
import scipy.io
from scipy.special import erf
from scipy.linalg import expm
from numpy import pi
from qutip import Qobj, mesolve
from itertools import combinations_with_replacement
from itertools import permutations
from itertools import product
import sys
class Lindblad:
    def __init__(self, H_op, A_op):
        self.H_op = H_op
        self.A_kraus_full = A_op
        self.Ns = H_op.shape[0]
        
    def Lindblad_exact(self,T,num_t,psi0):
        #generate exact by RK4
        num_t=num_t*10
        H = self.H_op
        K=len(self.A_kraus_full[0,0,:])
        Ns = psi0.shape[0]
        V_full=np.zeros((Ns,Ns,K),dtype=complex)
        for i in range(K):
            V_full[:,:,i]=self.A_kraus_full[:,:,i]
        tau = T / num_t
        rho = np.zeros((Ns,Ns,num_t+1), dtype=complex)
        rho[:,:,0]=np.outer(psi0,psi0.conj().T)
        for it in range(num_t):
            rho_mid=rho[:,:,it]
            rho_K=np.zeros((Ns,Ns,4), dtype=complex)
            rho_K[:,:,0]=-1j*(H@rho_mid-rho_mid@H)
            for i in range(K):
               rho_K[:,:,0]+=V_full[:,:,i]@rho_mid@V_full[:,:,i].conj().T\
                -1/2*(V_full[:,:,i].conj().T@V_full[:,:,i]@rho_mid+rho_mid@V_full[:,:,i].conj().T@V_full[:,:,i])
            rho_mid=rho[:,:,it]+tau/2*rho_K[:,:,0]
            rho_K[:,:,1]=-1j*(H@rho_mid-rho_mid@H)
            for i in range(K):
               rho_K[:,:,1]+=V_full[:,:,i]@rho_mid@V_full[:,:,i].conj().T\
                -1/2*(V_full[:,:,i].conj().T@V_full[:,:,i]@rho_mid+rho_mid@V_full[:,:,i].conj().T@V_full[:,:,i])
            rho_mid=rho[:,:,it]+tau/2*rho_K[:,:,1]
            rho_K[:,:,2]=-1j*(H@rho_mid-rho_mid@H)
            for i in range(K):
               rho_K[:,:,2]+=V_full[:,:,i]@rho_mid@V_full[:,:,i].conj().T\
                -1/2*(V_full[:,:,i].conj().T@V_full[:,:,i]@rho_mid+rho_mid@V_full[:,:,i].conj().T@V_full[:,:,i])
            rho_mid=rho[:,:,it]+tau*rho_K[:,:,2]
            rho_K[:,:,3]=-1j*(H@rho_mid-rho_mid@H)
            for i in range(K):
               rho_K[:,:,3]+=V_full[:,:,i]@rho_mid@V_full[:,:,i].conj().T\
                -1/2*(V_full[:,:,i].conj().T@V_full[:,:,i]@rho_mid+rho_mid@V_full[:,:,i].conj().T@V_full[:,:,i])
            rho[:,:,it+1]=rho[:,:,it]+tau*(rho_K[:,:,0]+2*rho_K[:,:,1]+2*rho_K[:,:,2]+rho_K[:,:,3])/6
        return rho
        
    def Lindblad_H(self, T, num_t, psi0, order): 
        #maximal is third order
        H = self.H_op
        K=len(self.A_kraus_full[0,0,:])
        Ns = psi0.shape[0]
        V_full=np.zeros((Ns,Ns,K),dtype=complex)
        for i in range(K):
            V_full[:,:,i]=self.A_kraus_full[:,:,i]
        tau = T / num_t
        Y_0=np.zeros((Ns,Ns,1,3),dtype=complex)
        Y_1=np.zeros((Ns,Ns,K,3),dtype=complex)
        Y_2=np.zeros((Ns,Ns,K,3),dtype=complex)
        Y_3=np.zeros((Ns,Ns,K,K,3),dtype=complex)
        Y_4=np.zeros((Ns,Ns,K,K,K,3),dtype=complex)
        X_0=np.zeros((Ns,Ns,1,3),dtype=complex)
        X_1=np.zeros((Ns,Ns,K,3),dtype=complex)
        X_2=np.zeros((Ns,Ns,K,3),dtype=complex)
        X_3=np.zeros((Ns,Ns,K,K,3),dtype=complex)
        X_4=np.zeros((Ns,Ns,K,K,K,3),dtype=complex)
        H_0=np.zeros((Ns,Ns,1),dtype=complex)
        H_1=np.zeros((Ns,Ns,K),dtype=complex)
        H_2=np.zeros((Ns,Ns,K),dtype=complex)
        H_3=np.zeros((Ns,Ns,K,K),dtype=complex)
        H_4=np.zeros((Ns,Ns,K,K,K),dtype=complex)
        G0=np.zeros((Ns,Ns),dtype=complex)
        for i in range(K):
            G0+=-1/2*V_full[:,:,i].conj().T@V_full[:,:,i]
        G=-1j*H+G0
        #F0= eye(size(G)) + G*dt + 1/2*G^2*dt^2+1/6*G^3*dt^3;
        Y_0[:,:,0,0]= G; Y_0[:,:,0,1]= G@G/2; Y_0[:,:,0,2]= G@G@G/6
        for i in range(K):
            V=V_full[:,:,i]
            #F1= -1i*( V + 1/2*V*G*dt+1/2*G*V*dt + (G^2*V+V*G^2+G*V*G)*dt^2/6 )*sqrt(dt);
            Y_1[:,:,i,0]=V; Y_1[:,:,i,1]= 1/2*(V@G+G@V); Y_1[:,:,i,2]= (G@G@V+V@G@G+G@V@G)/6
            #F2= -1i*(G*V - V*G)*dt*sqrt(dt)/sqrt(12);
            Y_2[:,:,i,1]= (G@V - V@G)/np.sqrt(12)
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            V_1=V_full[:,:,index[i][0]]
            V_2=V_full[:,:,index[i][1]]
            V_3=V_full[:,:,index[i][2]]
            #F4= -1i*V_lV_mV_n*dt*sqrt(dt)/sqrt(6);
            Y_4[:,:,index[i][0],index[i][1],index[i][2],1]= V_1@V_2@V_3/np.sqrt(6)
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):
            #F3= -1i*(V_mV_n/2 +1/6*dt*(G*V_mV_n+V_mV_n*G+V_m*G*V_n))*sqrt(2)*dt;
            V_1=V_full[:,:,index[i][0]]
            V_2=V_full[:,:,index[i][1]]
            Y_3[:,:,index[i][0],index[i][1],0]= V_1@V_2/np.sqrt(2)
            Y_3[:,:,index[i][0],index[i][1],1]= np.sqrt(2)/6*(G@V_1@V_2+V_1@V_2@G+V_1@G@V_2)

        ##determine the coefficients of the Ham matrix
        #X_{j,0}=Y_{j,0} for j\geq 2
        for i in range(K):
            X_1[:,:,i,0] = Y_1[:,:,i,0]
            X_2[:,:,i,0] = Y_2[:,:,i,0]
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            X_4[:,:,index[i][0],index[i][1],index[i][2],0] = Y_4[:,:,index[i][0],index[i][1],index[i][2],0]
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):    
            X_3[:,:,index[i][0],index[i][1],0] = Y_3[:,:,index[i][0],index[i][1],0]
        #Q0=\sum^N_{j=1}X_0X_0
        Q0= -2*G0 #simplify form
        #X_{0,0}=iY_{0,0}+1j/2*Q0
        X_0[:,:,0,0] = 1j*Y_0[:,:,0,0]+1j/2*Q0
        #Z_1=-1j/2*X_{0,0}-1/6*Q_0
        Z1= -1j/2*X_0[:,:,0,0] -1/6*Q0 
        #X_{j,1}=Y_{j,1}-X_{j,0}Z_1
        for i in range(K):
            X_1[:,:,i,1] = Y_1[:,:,i,1]-X_1[:,:,i,0] @ Z1
            X_2[:,:,i,1] = Y_2[:,:,i,1]-X_2[:,:,i,0] @ Z1
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            X_4[:,:,index[i][0],index[i][1],index[i][2],1] = Y_4[:,:,index[i][0],index[i][1],index[i][2],1]\
            -X_4[:,:,index[i][0],index[i][1],index[i][2],0] @ Z1
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):    
            X_3[:,:,index[i][0],index[i][1],1] = Y_3[:,:,index[i][0],index[i][1],1]\
            -X_3[:,:,index[i][0],index[i][1],0] @ Z1
        #Q_1=\sum^N_{j=1} X_0X_1+X_1X_0+\sum_{j=N+1}X_0X_0
        Q1=np.zeros((Ns,Ns),dtype=complex)
        for i in range(K):
            Q1+=X_1[:,:,i,0].conj().T @ X_1[:,:,i,1]
            Q1+=X_2[:,:,i,0].conj().T @ X_2[:,:,i,1]
            Q1+=X_1[:,:,i,1].conj().T @ X_1[:,:,i,0]
            Q1+=X_2[:,:,i,1].conj().T @ X_2[:,:,i,0]
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            Q1+=X_4[:,:,index[i][0],index[i][1],index[i][2],0].conj().T @ X_4[:,:,index[i][0],index[i][1],index[i][2],1]
            Q1+=X_4[:,:,index[i][0],index[i][1],index[i][2],1].conj().T @ X_4[:,:,index[i][0],index[i][1],index[i][2],0]
        index=list(product(range(K),repeat=2))
        for i in range(len(index)): 
            Q1+=X_3[:,:,index[i][0],index[i][1],0].conj().T @ X_3[:,:,index[i][0],index[i][1],0] 
        #X_{0,1}=1j*Y_{0,1}+1j/2*(Q_1+X^2_{0,0})-1j/24*Q^2_0+1/6*({Q0,X_{0,0})
        X_0[:,:,0,1]= 1j*Y_0[:,:,0,1] + 1j/2*Q1 + 1j/2*X_0[:,:,0,0]@X_0[:,:,0,0] + 1/6*(Q0@X_0[:,:,0,0]+X_0[:,:,0,0]@Q0)-1j/24*Q0@Q0
        #Z2=-1j/2*X_{0,1}-1/6*X^2_{0,0}-1/6*Q_{1}+1j/24*{Q_0,X_{0,0}}+1/120*Q^2_0
        Z2= -1j/2*X_0[:,:,0,1]-1/6*X_0[:,:,0,0]@X_0[:,:,0,0]-1/6*Q1+1j/24*(Q0@X_0[:,:,0,0]+X_0[:,:,0,0]@Q0)+1/120*Q0@Q0
        #X_{j,2}=Y_{j,2}-X_{j,1}Z_1-X_{j,0}Z_2
        for i in range(K):
            X_1[:,:,i,2] = Y_1[:,:,i,2]-X_1[:,:,i,1] @ Z1-X_1[:,:,i,0] @ Z2
            X_2[:,:,i,2] = Y_2[:,:,i,2]-X_2[:,:,i,1] @ Z1-X_2[:,:,i,0] @ Z2
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            X_4[:,:,index[i][0],index[i][1],index[i][2],2] = Y_4[:,:,index[i][0],index[i][1],index[i][2],2]\
            -X_4[:,:,index[i][0],index[i][1],index[i][2],1] @ Z1-X_4[:,:,index[i][0],index[i][1],index[i][2],0] @ Z2
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):    
            X_3[:,:,index[i][0],index[i][1],2] = Y_3[:,:,index[i][0],index[i][1],2]\
            -X_3[:,:,index[i][0],index[i][1],1] @ Z1-X_3[:,:,index[i][0],index[i][1],0] @ Z2 
        #Q2=\sum_{j=N+1}{X_{j,1},X_{j,0}}+\sum^N_{j=1}{X_{j,2},X_{j,0}}+\sum^N_{j=1}X_{j,1}X_{j,1}
        Q2=np.zeros((Ns,Ns),dtype=complex)
        for i in range(K):
            Q2+=X_1[:,:,i,1].conj().T @ X_1[:,:,i,1]
            Q2+=X_2[:,:,i,1].conj().T @ X_2[:,:,i,1]
            Q2+=X_1[:,:,i,0].conj().T @ X_1[:,:,i,2]
            Q2+=X_2[:,:,i,0].conj().T @ X_2[:,:,i,2]
            Q2+=X_1[:,:,i,2].conj().T @ X_1[:,:,i,0]
            Q2+=X_2[:,:,i,2].conj().T @ X_2[:,:,i,0]
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            Q2+=X_4[:,:,index[i][0],index[i][1],index[i][2],1].conj().T @ X_4[:,:,index[i][0],index[i][1],index[i][2],1]
            Q2+=X_4[:,:,index[i][0],index[i][1],index[i][2],0].conj().T @ X_4[:,:,index[i][0],index[i][1],index[i][2],2]
            Q2+=X_4[:,:,index[i][0],index[i][1],index[i][2],2].conj().T @ X_4[:,:,index[i][0],index[i][1],index[i][2],0]
        index=list(product(range(K),repeat=2))
        for i in range(len(index)): 
            Q2+=X_3[:,:,index[i][0],index[i][1],1].conj().T @ X_3[:,:,index[i][0],index[i][1],0] 
            Q2+=X_3[:,:,index[i][0],index[i][1],0].conj().T @ X_3[:,:,index[i][0],index[i][1],1] 
#         print(Q2)
        #X_{0,2}=........
        X_0[:,:,0,2]= 1j/6*G@G@G + 1j/2*Q2\
            +1j/2*(X_0[:,:,0,0]@X_0[:,:,0,1]+X_0[:,:,0,1]@X_0[:,:,0,0]) + 1/6*X_0[:,:,0,0]@X_0[:,:,0,0]@X_0[:,:,0,0]\
            +1/6*(Q1@X_0[:,:,0,0]+X_0[:,:,0,0]@Q1+Q0@X_0[:,:,0,1]+X_0[:,:,0,1]@Q0)\
            -1j/24*(Q0@X_0[:,:,0,0]@X_0[:,:,0,0]+X_0[:,:,0,0]@Q0@X_0[:,:,0,0]+X_0[:,:,0,0]@X_0[:,:,0,0]@Q0+Q0@Q1+Q1@Q0)\
            -1/120*(Q0@Q0@X_0[:,:,0,0]+Q0@X_0[:,:,0,0]@Q0+X_0[:,:,0,0]@Q0@Q0)+1j/720*Q0@Q0@Q0;
        
        #Hamiltonian construction
        H_0[:,:,0]= np.sqrt(tau)*(X_0[:,:,0,0] + tau*X_0[:,:,0,1]*int(order>=2)+ tau*tau*X_0[:,:,0,2]*int(order>=3))
        for i in range(K):
            H_1[:,:,i] = X_1[:,:,i,0] + tau*X_1[:,:,i,1]*int(order>=2)+ tau*tau*X_1[:,:,i,2]*int(order>=3)
            H_2[:,:,i] = X_2[:,:,i,0] + tau*X_2[:,:,i,1]*int(order>=2)+ tau*tau*X_2[:,:,i,2]*int(order>=3)
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            H_4[:,:,index[i][0],index[i][1],index[i][2]] =X_4[:,:,index[i][0],index[i][1],index[i][2],0] \
            + tau*X_4[:,:,index[i][0],index[i][1],index[i][2],1]*int(order>=2)\
            + tau*tau*X_4[:,:,index[i][0],index[i][1],index[i][2],2]*int(order>=3)
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):
            H_3[:,:,index[i][0],index[i][1]] =X_3[:,:,index[i][0],index[i][1],0] \
            + tau*X_3[:,:,index[i][0],index[i][1],1]*int(order>=2)\
            + tau*tau*X_3[:,:,index[i][0],index[i][1],2]*int(order>=3)
            H_3[:,:,index[i][0],index[i][1]]=np.sqrt(tau)*H_3[:,:,index[i][0],index[i][1]]
        #Construct large H
        size_H=int((1+2*K+K**2+K**3)*Ns)
        H_large=np.zeros((size_H,size_H), dtype=complex)
        for i in range(K):
            H_large[((i+1)*Ns):((i+2)*Ns),:Ns]=H_1[:,:,i]
            H_large[((i+K+1)*Ns):((i+K+2)*Ns),:Ns]=H_2[:,:,i]
        index=list(product(range(K),repeat=3))
        for i in range(len(index)):
            H_large[((2*K+i+1)*Ns):((2*K+i+2)*Ns),:Ns]=H_4[:,:,index[i][0],index[i][1],index[i][2]] 
        index=list(product(range(K),repeat=2))
        for i in range(len(index)):
            H_large[((K**3+2*K+i+1)*Ns):((K**3+2*K+i+2)*Ns),:Ns]=H_3[:,:,index[i][0],index[i][1]]     
        H_large=H_large+H_large.conj().T
        H_large[:Ns,:Ns]=H_0[:,:,0]
        E_H, V_H = la.eigh(H_large) 
        print('H_0-H^\dag_0=',la.norm(H_0[:,:,0]-H_0[:,:,0].conj().T))
        exp_H=V_H @ la.expm(-1j*np.sqrt(tau)*np.diag(E_H)) @ V_H.conj().T
        rho = np.zeros((Ns,Ns,num_t+1), dtype=complex)
        rho[:,:,0]=np.outer(psi0,psi0.conj().T)
        for it in range(num_t):
            ancilla=np.zeros(1+2*K+K**2+K**3)
            ancilla[0]=1
            rho_large=np.kron(np.diag(ancilla),rho[:,:,it])
            rho_large=exp_H@rho_large@exp_H.conj().T
            for i in range(1+2*K+K**2+K**3):
               rho[:,:,it+1]+=rho_large[i*Ns:(i+1)*Ns,i*Ns:(i+1)*Ns]
        return rho, H_large
    
#     def Lindblad_SDE_third(self,T,num_t,psi0):
#         H = self.H_op
#         K=len(self.A_kraus_full[0,0,:])
#         Ns = psi0.shape[0]
#         V_full=np.zeros((Ns,Ns,K),dtype=complex)
#         for i in range(K):
#             V_full[:,:,i]=self.A_kraus_full[:,:,i]
#         tau = T / num_t
#         F_0=np.zeros((Ns,Ns,1),dtype=complex)
#         F_1=np.zeros((Ns,Ns,K),dtype=complex)
#         F_2=np.zeros((Ns,Ns,K),dtype=complex)
#         F_3=np.zeros((Ns,Ns,K,K),dtype=complex)
#         F_4=np.zeros((Ns,Ns,K,K,K),dtype=complex)
#         G0=np.zeros((Ns,Ns),dtype=complex)
#         for i in range(K):
#            G0+=-1/2*V_full[:,:,i].conj().T@V_full[:,:,i]
#         G=-1j*H+G0
#         F_0[:,:,0]=np.eye(Ns,dtype=complex)+G*tau+1/2*G@G*tau**2+1/6*G@G@G*tau**3
#         for i in range(K):
#            V=V_full[:,:,i]
#            F_1[:,:,i]=np.sqrt(tau)*(V+tau/2*(V@G+G@V)+tau**2/6*(G@G@V+G@V@G+V@G@G))
#            F_2[:,:,i]=np.sqrt(tau)*tau/np.sqrt(12)*(G@V-V@G)
#         index=list(product(range(K),repeat=2))
#         for i in range(len(index)):
#            V_1=V_full[:,:,index[i][0]]
#            V_2=V_full[:,:,index[i][1]]
#            F_3[:,:,index[i][0],index[i][1]]=np.sqrt(2)*tau*(1/2*V_1@V_2+tau/6*(G@V_1@V_2+V_1@G@V_2+V_1@V_2@G))
#         index=list(product(range(K),repeat=3))
#         for i in range(len(index)):
#            V_1=V_full[:,:,index[i][0]]
#            V_2=V_full[:,:,index[i][1]]
#            V_3=V_full[:,:,index[i][2]]
#            F_4[:,:,index[i][0],index[i][1],index[i][2]]=np.sqrt(tau)*tau/np.sqrt(6)*V_1@V_2@V_3 
#         rho = np.zeros((Ns,Ns,num_t+1), dtype=complex)
#         rho[:,:,0]=np.outer(psi0,psi0.conj().T)
#         for it in range(num_t):
#             rho_mid=F_0[:,:,0]@rho[:,:,it]@F_0[:,:,0].conj().T
#             for i in range(K):
#                rho_mid+=F_1[:,:,i]@rho[:,:,it]@F_1[:,:,i].conj().T
#                rho_mid+=F_2[:,:,i]@rho[:,:,it]@F_2[:,:,i].conj().T
#             index=list(product(range(K),repeat=2))
#             for i in range(len(index)):                                 
#                rho_mid+=F_3[:,:,index[i][0],index[i][1]]@rho[:,:,it]@F_3[:,:,index[i][0],index[i][1]].conj().T
#             index=list(product(range(K),repeat=3))
#             for i in range(len(index)):
#                rho_mid+=F_4[:,:,index[i][0],index[i][1],index[i][2]]@rho[:,:,it]@F_4[:,:,index[i][0],index[i][1],index[i][2]].conj().T
#             rho[:,:,it+1]=rho_mid
#         return rho

        
       
    

