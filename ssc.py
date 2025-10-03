from functions import Kinematics, RKinematics, Thermodynamics, PlotTS, PlotHS, Radial, BladeLosses, Efficiency
from classes import VelocityTriangle, TState ,StageGeometry ,BladeGeometry
import numpy as np
import warnings
from scipy.optimize import fsolve, brentq
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP
from scipy.interpolate import griddata
from collections import defaultdict


z           = 1                                            
n           = 18000
p1t, T1t    = 333162.0, 376.0
fluid       = 'hydrogen'
eta, etaR   = 0.85, 0.85
# phi, psi, r = 0.6, 0.4, 0.8
phi, psi, r = 0.53, 0.38, 0.8163082732946852
# phi, psi, r = 0.45, 0.5, 0.6
# phi, psi, r = 0.5514016026103697, 0.3607428090407071, 0.5724153500647056


m           = 20.0
dh          = 55e3
R           = CP.PropsSI("GAS_CONSTANT", fluid)/CP.PropsSI("MOLAR_MASS", fluid)
Cp          = CP.PropsSI("CPMASS", "P", p1t, "T", T1t, fluid)
Cv          = CP.PropsSI("CVMASS", "P", p1t, "T", T1t, fluid)
gamma       = Cp / Cv

IC=np.zeros(18)
IC[1], IC[4] = p1t, T1t   
TS = TState(IC)

iteration, threshold, max_iter = 0, 1e-6, 1000 

while iteration < max_iter:
    M1 = 0.9
    p3 = p1t * (1 + dh / (Cp * T1t * eta)) ** (gamma / (gamma - 1))
    p1 = p1t * (1 + (gamma - 1)/2 * M1**2) ** (-gamma/(gamma - 1))
    T1 = T1t / (1 + (gamma - 1)/2 * M1**2)
    a1 = CP.PropsSI("A", "P", p1, "T", T1, fluid)
    c3 = c1 = a1 * M1

    # u = np.sqrt(dh / psi)
    u  = np.sqrt((dh + (c1**2 / 2)) / (psi + 0.5 * (phi**2 + (1 - r - psi/2)**2)))
    Dm = 60 * u / (np.pi * n)

    VT  = VelocityTriangle(*Kinematics(psi, phi, r, u))
    TS1, TS2, TS3 = map(TState, Thermodynamics(VT, TS, p3, fluid, eta, etaR))
    STAGE_GEOMETRY  = StageGeometry(VT, TS1, TS2, TS3, m, Dm)    
    ROTOR_GEOMETRY  = BladeGeometry(VT, STAGE_GEOMETRY, TS1, TS2, TS3, Dm, True)
    STATOR_GEOMETRY = BladeGeometry(VT, STAGE_GEOMETRY, TS1, TS2, TS3, Dm, False)
    
    for geom, state1, state2 in [(STATOR_GEOMETRY, TS2, TS3),(ROTOR_GEOMETRY, TS1, TS2),]:
        Radial(phi, psi, r, Dm, geom.Dt, u)
        Radial(phi, psi, r, Dm, geom.Dh, u)
        
    YR = BladeLosses(TS1, TS2, ROTOR_GEOMETRY, VT, True)
    YS = BladeLosses(TS2, TS3, STATOR_GEOMETRY, VT, False)

    etaiter, etaRiter, etaTT, etaTS = Efficiency(VT, TS1, TS2, TS3, YR, YS, fluid)

    if abs(etaRiter - etaR) / etaRiter < threshold and abs(etaiter - eta) / etaiter < threshold:
        eta, etaR = etaiter, etaRiter
        M1 = TS1.MR
        break

    eta, etaR, M1 = etaiter, etaRiter, TS1.MR
    iteration += 1     
    
    
PlotHS(TS1, TS2, TS3)
PlotTS(TS1, TS2, TS3)
VT.plotVT()

for geom in [ROTOR_GEOMETRY, STATOR_GEOMETRY]:
    print(f"a1: {np.rad2deg(geom.a1):.2f}°, a2: {np.rad2deg(geom.a2):.2f}°, stagg: {np.rad2deg(geom.stagg):.2f}°")


# === Off-Design Initialization ===
p1t_design, T1t_design, m_design = 333162.0, 376.0, 20
Tstd, Pstd = 288.15, 101325
design_rpm = 18000 #* np.sqrt(T1t_design / Tstd)

R = CP.PropsSI("GAS_CONSTANT", fluid)/CP.PropsSI("MOLAR_MASS", fluid)

rpm_range = np.linspace(0.6 * design_rpm, 1.15 * design_rpm, 7)
m_off_range = np.linspace(0.6 * m_design, 1.5 * m_design, 300)

Cp_off = CP.PropsSI("CPMASS", "P", p1t_design, "T", T1t_design, fluid)
Cv_off = CP.PropsSI("CVMASS", "P", p1t_design, "T", T1t_design, fluid)
gamma_off = Cp_off / Cv_off

results = []

#relative tip mach number control


def compute_mach(m_off, p1t, T1t, gamma, A1):

    if m_off <= 0:
        return 0.0

    def G(M):
        return M * (1.0 + (gamma - 1.0) / 2.0 * M**2) ** (-(gamma + 1.0) / (2.0 * (gamma - 1.0)))

    coeff = A1 * p1t / np.sqrt(T1t) * np.sqrt(gamma / R)
    m_choke = coeff * G(1.0)

    if m_off >= m_choke:
        return 1.0

    def f(M):
        return coeff * G(M) - m_off

    M = brentq(f, 1e-8, 1.0 - 1e-8)
    return float(M)

def compute_flow(rpm, m_off):
    
    try:
        M1_off = compute_mach(m_off, p1t_design, T1t_design, gamma_off, STAGE_GEOMETRY.A1)
    except Exception as e:
        warnings.warn(f"Mach solve failed at RPM={rpm:.0f}, m_off={m_off:.6f}: {e}")
        return None
    
    m_corr   = m_off * np.sqrt(T1t_design / Tstd) / (p1t_design / Pstd)
    etaR_off, eta_off = 0.85, 0.85
    
    IC_off = np.zeros(18)
    IC_off[1], IC_off[4] = p1t_design, T1t_design
    TS_off = TState(IC_off)

    for _ in range(100):
        try:
            M1_off = compute_mach(m_off, p1t_design, T1t_design, gamma_off, STAGE_GEOMETRY.A1)
            p1_off = p1t_design * (1 + (gamma_off - 1) / 2 * M1_off ** 2) ** (-gamma_off / (gamma_off - 1))
            T1_off = T1t_design / (1 + (gamma_off - 1) / 2 * M1_off ** 2)
            a1_off = CP.PropsSI("A", "P", p1_off, "T", T1_off, fluid)
            c1_off = a1_off * M1_off
            w1_off = c1_off / np.cos(ROTOR_GEOMETRY.a1)
            
            # b1_off = RKinematics.a1
            # b2_off = RKinematics.a2
            
            rho1_off = CP.PropsSI("D", "P", p1_off, "T", T1_off, fluid)
            inc    = ROTOR_GEOMETRY.inc
            dev    = ROTOR_GEOMETRY.deltaw
            if   hasattr(ROTOR_GEOMETRY, "b1B"): beta1B = ROTOR_GEOMETRY.b1B
            elif hasattr(ROTOR_GEOMETRY, "a1B"): beta1B = ROTOR_GEOMETRY.a1B
            else: raise AttributeError("ROTOR_GEOMETRY has neither b1B nor a1B.")
            
            if   hasattr(ROTOR_GEOMETRY, "b2B"): beta2B = ROTOR_GEOMETRY.b2B
            elif hasattr(ROTOR_GEOMETRY, "a2B"): beta2B = ROTOR_GEOMETRY.a2B
            else: raise AttributeError("ROTOR_GEOMETRY has neither b2B nor a2B.")

            
            u_off = Dm * rpm * np.pi / 60 
            # VT_off = VelocityTriangle(*RKinematics(w1_off, b1_off, b2_off, u_off, r))
            # VT_off = VelocityTriangle(*RKinematics(
            #     m_off, STAGE_GEOMETRY.A1, u_off, r, rho1_off, beta1B, beta2B, inc, dev
            # ))
            VT_off = VelocityTriangle(*RKinematics(
                m_off,
                STAGE_GEOMETRY.A1,
                u_off,
                rho1_off,
                r,        # NEW: absolute α1 at rotor inlet
                beta1B,            # rotor metal β1B
                beta2B,            # rotor metal β2B
                inc,               # rotor incidence i_R (from bladegeometry, rotor)
                dev,               # rotor deviation δ_R (from bladegeometry, rotor)
            ))
            
            dh_off = u_off**2 * (VT_off.psi + 0.5*(VT_off.phi**2 + 
                                                   (1 - VT_off.r - 0.5*VT_off.psi)**2)) - 0.5*VT_off.c1**2
          
            p3_off = p1t_design * (1 + dh_off / (Cp_off * T1t_design * eta_off)) ** (gamma_off / (gamma_off - 1))
            TS1_off, TS2_off, TS3_off = map(TState, Thermodynamics(VT_off, TS_off, p3_off, fluid, eta_off, etaR_off))
            
            YR_off = BladeLosses(TS1_off, TS2_off, ROTOR_GEOMETRY, VT_off, True)
            YS_off = BladeLosses(TS2_off, TS3_off, STATOR_GEOMETRY, VT_off, False)

            etaiter_off, etaRiter_off, etaTT_off, etaTS_off = Efficiency(
                VT_off, TS1_off, TS2_off, TS3_off, YR_off, YS_off, fluid
            )
            
    
            if abs(etaRiter_off - etaR_off) / etaRiter_off < 1e-4 and abs(etaiter_off - eta_off) / etaiter_off < 1e-4:
                return {
                    'rpm': rpm,
                    'm_corr': m_corr,
                    'eta_off': etaiter_off,
                    'etaR_off': etaRiter_off,
                    'etaTT_off': etaTT_off,
                    'etaTS_off': etaTS_off,
                    'TS1': TS1_off,
                    'TS2': TS2_off,
                    'TS3': TS3_off,
                    'M1_off': M1_off,
                    'VT': VT_off
                }

            eta_off, etaR_off, M1_off = etaiter_off, etaRiter_off, TS1_off.MR
            
            
        except Exception as e:
            warnings.warn(f"Iteration failed at RPM={rpm:.0f}, m_off={m_off:.6f}: {e}")
            return None
        
    return None

# === Run Performance Map ===
for rpm in rpm_range:
    for m_off in m_off_range:
        print(f"RPM = {rpm:.0f}, m_off = {m_off:.6f}")
        result = compute_flow(rpm, m_off)
        if result:
            results.append(result)
            print(f"m_corr = {result['m_corr']:.6f}")
        else:
            warnings.warn(f"Computation skipped at RPM={rpm:.0f}, m_off={m_off:.6f}")
            
# === Organize and Plot Results ===
rpm_groups = defaultdict(list)
x_vals, y_vals, z_vals = [], [], []

for res in results:

    scaled_m_corr = res['m_corr']
    pressure_ratio = res['TS3'].pt / p1t_design
    eta_off = res['eta_off']

    x_vals.append(scaled_m_corr)
    y_vals.append(pressure_ratio)
    z_vals.append(eta_off)
    rpm_groups[res['rpm']].append((scaled_m_corr, pressure_ratio))

xi = np.linspace(min(x_vals), max(x_vals), 500)
yi = np.linspace(min(y_vals), max(y_vals), 500)
xi, yi = np.meshgrid(xi, yi)
zi = griddata((x_vals, y_vals), z_vals, (xi, yi), method='cubic')
plt.figure(figsize=(6, 4))

# Contour plot with color map
cp = plt.contourf(xi, yi, zi, levels=300, cmap='RdYlGn')
plt.colorbar(cp, label='Isentropic Efficiency')

# Plot RPM lines in black with percentage labels
for rpm, data in sorted(rpm_groups.items()):
    data.sort()
    m_corr_values, pressure_ratios_ = zip(*data)
    plt.plot(m_corr_values, pressure_ratios_, color='black', linewidth=0.8)

    # Add inline RPM percentage label
    right_idx = -1  # index of last element
    rpm_percent = round(rpm / 18000 * 100)
    plt.text(
        m_corr_values[right_idx],
        pressure_ratios_[right_idx],
        f'{rpm_percent}%',
        fontsize=7,
        ha='right',
        va='top',
        color='black'
    )

plt.xlabel('Reduced Mass Flow Rate [kg/s]')
plt.ylabel('Pressure Ratio')
plt.grid(True)
plt.tight_layout()
plt.xlim(min(x_vals) * 0.98, max(x_vals) * 1.016)  # 10% extra space on both sides
plt.ylim(min(y_vals) * 0.98, max(y_vals) * 1.016)
plt.savefig('off_isentropic_efficiency.pdf', format='pdf')
plt.show()

