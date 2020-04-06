gamma = 1.4     #Гамма, она же каппа
#Различные формулы из гаммы
g_m_1_over_2g = (gamma-1)/2/gamma      #g1
g_p_1_over_2g = (gamma+1)/2/gamma      #g2
g_m_1_over_2g_inv = 1/g_m_1_over_2g    #g3
g_m_1_over_2_inv = 2/(gamma-1)         #g4
g_p_1_over_2_inv = 2/(gamma+1)         #g5
g_m_1_over_g_p_1 = (gamma-1)/(gamma+1) #g6
g_m_1_over_2 = (gamma-1)/2             #g7
g_m_1 = gamma-1  

tol = 1e-8

import numpy as np

def sound_speed(W):
    d = W[0]
    p = W[2]
    return (gamma*(p/d))**0.5

def enthalpy(W):
    return 0.5*W[1]**2 + gamma/ g_m_1 *(W[2]/W[0])

def U_to_W(U):
    W = np.zeros_like(U)
    W[0] = U[0]
    W[1] = U[1]/U[0]
    W[2] = g_m_1*(U[2] - 0.5*U[1]**2/U[0])
    return W


def W_to_U(W):
    U = np.zeros_like(W)
    U[0] = W[0]
    U[1] = W[1]*W[0]
    U[2] = 0.5*W[1]**2*W[0]+W[2]/ g_m_1
    return U


def flux(W):
    F = np.zeros_like(W)
    F[0] = W[1]*W[0]
    F[1] = W[1]**2*W[0] + W[2]
    F[2] = W[1]*(0.5*W[1]**2*W[0]+W[2]/ g_m_1 + W[2])
    return F


def roe_average(Wl,Wr):
    denom = Wl[0]**0.5 + Wr[0]**0.5
    u_tilda = (Wl[0]**0.5 * Wl[1] + Wr[0]**0.5 * Wr[1])/ denom
    H_tilda = (Wl[0]**0.5 * enthalpy(Wl) + Wr[0]**0.5 * enthalpy(Wr))/ denom
    a_tilda = (g_m_1*(H_tilda - 0.5*u_tilda**2))**0.5
    return u_tilda, H_tilda, a_tilda 

def roe_eigen_values(roe_average):
    lambdas = np.zeros(3)
    lambdas[0] = roe_average[0] - roe_average[2]
    lambdas[1] = roe_average[0] 
    lambdas[2] = roe_average[0] + roe_average[2]
    return lambdas

def roe_eigen_vectors(roe_average):
    K = np.zeros((3,3))
    K[0] = np.asarray([1, 
                       roe_average[0] - roe_average[2], 
                       roe_average[1] - roe_average[0] * roe_average[2] ])
    K[1] = np.asarray([1, 
                       roe_average[0] , 
                       0.5*roe_average[0] **2 ])
    K[2] = np.asarray([1, 
                       roe_average[0] + roe_average[2], 
                       roe_average[1] + roe_average[0] * roe_average[2] ])
    return K


def alfas(dU, roe_average):
    alfas = np.zeros(3)
    alfas[1] = g_m_1/roe_average[2]**2*(dU[0]*(roe_average[1] - roe_average[0]**2) + dU[1]*roe_average[0] - dU[2] )
    alfas[0] = 0.5/roe_average[2]*(dU[0]*(roe_average[0] + roe_average[2]) - dU[1] - roe_average[2]*alfas[1])
    alfas[2] = dU[0] - (alfas[0] +alfas[1])
    return alfas