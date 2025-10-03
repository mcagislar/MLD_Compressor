import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import CoolProp.CoolProp as CP

from functions import Kinematics, Thermodynamics, BladeLosses, Efficiency
from classes import VelocityTriangle, TState, StageGeometry, BladeGeometry


# --- Logging Lists ---
L_phi, L_psi, L_r     = [], [], []
L_n, L_eta, L_u, L_Dm = [], [], [], []

LAMBDA = 25.0  # başlangıç için; sonra kalibre et

# --- Objective Function ---
def function(X):
    phi, psi, r = X
    n = 18000
    fluid = 'hydrogen'
    eta = etaR = 0.85
    m = 20.0
    dh = 24e4
    p1t = 333162.0
    T1t = 376.0
    
    try:
        Cp = CP.PropsSI("CPMASS", "P", p1t, "T", T1t, fluid)
        Cv = CP.PropsSI("CVMASS", "P", p1t, "T", T1t, fluid)
        gamma = Cp / Cv

        IC = np.zeros(18)
        IC[1], IC[4] = p1t, T1t
        TS = TState(IC)

        threshold = 1e-6
        max_iter = 1000

        for _ in range(max_iter):
            M1 = 0.7
            p3 = p1t * (1 + dh / (Cp * T1t * eta)) ** (gamma / (gamma - 1))
            p1 = p1t * (1 + (gamma - 1)/2 * M1**2) ** (-gamma/(gamma - 1))
            T1 = T1t / (1 + (gamma - 1)/2 * M1**2)
            a1 = CP.PropsSI("A", "P", p1, "T", T1, fluid)
            c1 = a1 * M1
            
            u = np.sqrt(dh / psi)
            # u  = np.sqrt((dh + (c1**2 / 2)) / (psi + 0.5 * (phi**2 + (1 - r - psi/2)**2)))
            Dm = 60 * u / (np.pi * n)

            VT = VelocityTriangle(*Kinematics(psi, phi, r, u))
            TS1, TS2, TS3 = [TState(x) for x in Thermodynamics(VT, TS, p3, fluid, eta, etaR)]
            STAGE_GEOMETRY = StageGeometry(VT, TS1, TS2, TS3, m, Dm)
            ROTOR_GEOMETRY = BladeGeometry(VT, STAGE_GEOMETRY, TS1, TS2, TS3, Dm, True)
            STATOR_GEOMETRY = BladeGeometry(VT, STAGE_GEOMETRY, TS1, TS2, TS3, Dm, False)

            YR = BladeLosses(TS1, TS2, ROTOR_GEOMETRY, VT, True)
            YS = BladeLosses(TS2, TS3, STATOR_GEOMETRY, VT, False)

            etaR_iter, eta_iter, etaTT, etaTS = Efficiency(VT, TS1, TS2, TS3, YR, YS, fluid)

            if abs(etaR_iter - etaR) / etaR_iter < threshold and abs(eta_iter - eta) / eta_iter < threshold:
                eta, etaR, M1 = eta_iter, etaR_iter, TS1.MR
                break

            eta, etaR, M1 = eta_iter, etaR_iter, TS1.MR
            
        # Logging values
        L_phi.append(phi)
        L_psi.append(psi)
        L_r.append(r)
        L_n.append(n)
        L_eta.append(eta)
        L_u.append(u)
        L_Dm.append(Dm)


        # Penalty conditions
        if eta > 0.99 or etaR > 0.99 or etaTT > 0.99 or etaTS > 0.99 or YR < 0 or YS < 0:
            return 1e9
        if abs(VT.w2 / VT.w1) <= 0.72:
            return 1e9
        if abs(VT.c2 / VT.c1) <= 0.72:
            return 1e9
        if abs(VT.c1 / np.cos(VT.b1)) != VT.w1:
            return 1e9

        return 1 / eta

    except Exception as e:
        print(f"Error encountered: {e}, returning penalty value.")
        return 1e9


# --- Optimization ---
start_time = time.time()

bounds = [(0.1, 0.7), (0.1, 0.45), (0.1, 1)]
result = differential_evolution(function, bounds)
end_time = time.time()

execution_time = end_time - start_time
print(result)
print(f"Execution Time: {execution_time:.2f} seconds")

n_err = result.nfev - len(L_phi)
print(f"Number of errors encountered: {n_err}")
result.nfev = len(L_phi)
    
# --- Plotting ---
# Efficiency plot
plt.figure(figsize=(6, 4))
scatter = plt.scatter(range(result.nfev), L_eta, c=L_eta, cmap='RdYlGn', edgecolor='k', vmax=1.0, vmin=0.0)
plt.xlabel('Iterations')
plt.ylabel('Isentropic Efficiency')
plt.ylim(0,1)
plt.grid(True)
plt.savefig('efficiency.pdf', format='pdf')

# Flow coefficient plot
plt.figure(figsize=(6, 4))
scatter = plt.scatter(range(result.nfev), L_phi, c=L_eta, cmap='RdYlGn', edgecolor='k')
plt.xlabel('Iterations')
plt.ylabel('Flow coefficient')
plt.colorbar(scatter, label='Isentropic Efficiency')
plt.grid(True)
plt.savefig('flow.pdf', format='pdf')

# Work coefficient plot
plt.figure(figsize=(6, 4))
scatter = plt.scatter(range(result.nfev), L_psi, c=L_eta, cmap='RdYlGn', edgecolor='k')
plt.xlabel('Iterations')
plt.ylabel('Work coefficient')
plt.colorbar(scatter, label='Isentropic Efficiency')
plt.grid(True)
plt.savefig('work.pdf', format='pdf')

# Degree of reaction plot
plt.figure(figsize=(6, 4))
scatter = plt.scatter(range(result.nfev), L_r, c=L_eta, cmap='RdYlGn', edgecolor='k')
plt.xlabel('Iterations')
plt.ylabel('Degree of reaction')
plt.colorbar(scatter, label='Isentropic Efficiency')
plt.grid(True)
plt.savefig('reaction.pdf', format='pdf')

# Mean diameter plot
plt.figure(figsize=(6, 4))
scatter = plt.scatter(range(result.nfev), L_Dm, c=L_eta, cmap='RdYlGn', edgecolor='k')
plt.xlabel('Iterations')
plt.ylabel('Mean diameter [m]')
plt.colorbar(scatter, label='Isentropic Efficiency')
plt.grid(True)
plt.savefig('diameter.pdf', format='pdf')


