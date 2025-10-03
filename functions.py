import numpy as np  # Numerical operations
import matplotlib.pyplot as plt  # Plotting utilities
import CoolProp.CoolProp as CP  # Thermophysical properties
from typing import Any
from dataclasses import dataclass


def Kinematics(psi, phi, r, u):
 
    a1 = 0
    # a1 = np.arctan((1 - r - psi/2) / phi)
    # if a1<0: a1 = 0
    a2 = np.arctan((1 - r + psi/2) / phi)
    b1 = np.arctan((r + psi/2) / phi)      
    b2 = np.arctan((r - psi/2) / phi)
    
    cm = u * phi
    c1_tangential = cm * np.tan(a1)
    c2_tangential = cm * np.tan(a2)
    w1_tangential = cm * np.tan(b1)
    w2_tangential = cm * np.tan(b2)
    
    c1 = np.hypot(cm, c1_tangential)
    c2 = np.hypot(cm, c2_tangential)
    w1 = np.hypot(cm, w1_tangential)
    w2 = np.hypot(cm, w2_tangential)
    
    angles = [a1, a2, b1, b2] 
    v_abs  = [cm, c1, c2, w1, w2]
    v_tg   = [c1_tangential, c2_tangential, w1_tangential, w2_tangential]

    return psi, phi, r, u, angles, v_abs, v_tg

def RKinematics(
    m, A1, u,
    rho1,
    r,             # actual α1 at rotor inlet (already includes stator i/δ)
    beta1_metal,        # rotor metal angles
    beta2_metal,
    inc,              # rotor incidence  i_R  (same sign convention as bladegeometry)
    dev,              # rotor deviation  δ_R  (same convention as bladegeometry)
    A2=None, rho2=None  # optional for cm2; if None, assume cm2≈cm1
):
    # 1) Meridional components from continuity
    cm1 = m / (A1 * rho1)
    cm2 = m / (A2 * rho2) if (A2 and rho2) else cm1

    # 2) Actual rotor relative angles from metal + i/δ  (match your bladegeometry signs!)
    b1 = beta1_metal + inc          # β1 = β1B + i_R
    b2 = beta2_metal + dev          # β2 = β2B + δ_R

    # 3) Relative tangential components
    w1u = cm1 * np.tan(b1)
    w2u = cm2 * np.tan(b2)

    # 4) Compressor absolute tangential components (NOTE THE SIGN)
    c1u = u - w1u
    c2u = u - w2u

    # 5) Absolute exit angle and velocity magnitudes
    a1 = np.arctan2(c1u, cm1)
    a2 = np.arctan2(c2u, cm2)

    c1 = np.hypot(cm1, c1u); c2 = np.hypot(cm2, c2u)
    w1 = np.hypot(cm1, w1u); w2 = np.hypot(cm2, w2u)

    # 6) Non-dimensionals (local)
    phi1 = cm1 / u; phi2 = cm2 / u
    psi  = (c2u - c1u) / u                 # compressor ψ

    # 7) Degree of reaction (use compressor form; prefer local φ2)
    R = 1 - phi2 * np.tan(a2) + 0.5 * psi

    angles = [a1, a2, b1, b2]
    v_abs  = [cm1, c1, c2, w1, w2]
    v_tg   = [c1u, c2u, w1u, w2u]
    return psi, 0.5*(phi1+phi2), R, u, angles, v_abs, v_tg

# def RKinematics(
#     m,                   # mass flow [kg/s]
#     A1,                  # inlet area [m^2]
#     u,                   # blade speed at Dm [m/s]
#     r,                   # degree of reaction (pass-through here)
#     rho1,                # inlet density [kg/m^3]
#     beta1_metal,         # β1B [rad]
#     beta2_metal,         # β2B [rad]
#     inc,                 # i  [rad],  positive if β1 > β1B
#     dev,                 # δ  [rad],  positive if β2B > β2
#     A2=None, rho2=None   # optional outlet area/density if available
# ):
#     # 1) Meridional components from continuity
#     cm1 = m / (A1 * rho1)
#     cm2 = m / (A2 * rho2) if (A2 and rho2) else cm1

#     # Actual relative angles from metal + incidence/deviation
#     b1 = beta1_metal + inc       # β1 = β1B + i
#     b2 = beta2_metal + dev       # β2 = β2B + δ

#     # Relative tangential components
#     w1u = cm1 * np.tan(b1)
#     w2u = cm2 * np.tan(b2)

#     # Absolute tangential components (c_u = w_u + u)
#     c1u = w1u + u
#     c2u = w2u + u

#     # Absolute flow angles
#     a1 = np.arctan2(c1u, cm1)          # α1
#     a2 = np.arctan2(c2u, cm2)          # α2

#     # Velocity magnitudes
#     c1 = np.hypot(cm1, c1u)
#     c2 = np.hypot(cm2, c2u)
#     w1 = np.hypot(cm1, w1u)
#     w2 = np.hypot(cm2, w2u)

#     # Non-dimensional coefficients (common definitions)
#     phi = 0.5 * (cm1 + cm2) / u        # flow coefficient (avg)
#     psi = (c2u - c1u) / u              # loading coefficient (Euler / U^2)
#     # psi = phi * np.tan(b1 - b2)
    
#     angles = [a1, a2, b1, b2]
#     v_abs  = [cm1, c1, c2, w1, w2]
#     v_tg   = [c1u, c2u, w1u, w2u]

#     return psi, phi, None, u, angles, v_abs, v_tg

# def RKinematics(cm, b1, b2, u, r):
    
#     w1_tg = cm * np.tan(b1)
#     c2_tg = cm * np.tan(b2)
#     # w2_tg = cm * np.tan(b2)
#     w2_tg = c2_tg - u
#     # c2_tg = w2_tg - u
#     c1_tg = w1_tg - u
    
#     a1 = np.arctan(c1_tg / cm)
#     a2 = np.arctan(c2_tg / cm)
#     # b2 = np.arctan(w2_tg / cm)

#     c1 = np.hypot(cm, c1_tg)
#     c2 = np.hypot(cm, c2_tg)
#     w1 = np.hypot(cm, w1_tg)
#     w2 = np.hypot(cm, w2_tg)

#     angles = [a1, a2, b1, b2]
#     v_abs  = [cm, c1, c2, w1, w2]
#     v_tg   = [c1_tg, c2_tg, w1_tg, w2_tg]

#     phi = cm/u
#     # psi = (c2_tg - c1_tg)/u
#     psi = phi * np.tan(b1 - b2) #######
#     # psi = 1 - 0.438*phi     #####
#     # r = phi * 0.5 * (np.tan(b1) + np.tan(b2))
#     # r = 0.5 * np.tan(b1 + b2) ##########

#     return phi, psi, r, u, angles, v_abs, v_tg

def Thermodynamics(VelocityTriangle, TState, p3, fluid, eta, etaR):
    p1t, T1t  = TState.pt, TState.Tt
    c1, c2    = VelocityTriangle.c1, VelocityTriangle.c2
    w1, w2    = VelocityTriangle.w1, VelocityTriangle.w2

    # ---------- Rotor inlet (1) ----------
    h1t = CP.PropsSI("H", "P", p1t, "T", T1t, fluid)
    s1  = CP.PropsSI("S", "P", p1t, "T", T1t, fluid)

    h1 = h1t - 0.5 * c1**2
    p1 = CP.PropsSI("P", "H", h1, "S", s1, fluid)
    T1 = CP.PropsSI("T", "H", h1, "S", s1, fluid)
    d1 = CP.PropsSI("D", "P", p1, "T", T1, fluid)
    d1t = CP.PropsSI("D", "H", h1t, "S", s1, fluid)
    
    h1tr = h1 + 0.5 * w1**2
    p1tr = CP.PropsSI("P", "H", h1tr, "S", s1, fluid)
    T1tr = CP.PropsSI("T", "H", h1tr, "S", s1, fluid)

    # ---------- Rotor outlet (2) ----------
    h2tr = h1tr
    h2  = h2tr - 0.5 * w2**2
    delta_hR = h2 - h1
    delta_hR_is = etaR * delta_hR
    h2is = h1 + delta_hR_is

    p2 = CP.PropsSI("P", "S", s1, "H", h2is, fluid)
    s2 = CP.PropsSI("S", "P", p2, "H", h2, fluid)
    T2 = CP.PropsSI("T", "P", p2, "H", h2, fluid)
    d2 = CP.PropsSI("D", "P", p2, "H", h2, fluid)

    h2t  = h2 + 0.5 * c2**2
    p2t  = CP.PropsSI("P", "H", h2t, "S", s2, fluid)
    T2t  = CP.PropsSI("T", "H", h2t, "S", s2, fluid)
    d2t  = CP.PropsSI("D", "H", h2t, "S", s2, fluid)

    p2tr = CP.PropsSI("P", "H", h2tr, "S", s2, fluid)
    T2tr = CP.PropsSI("T", "H", h2tr, "S", s2, fluid)

    # ---------- Stator outlet (3) ----------
    h3is = CP.PropsSI("H", "P", p3, "S", s1, fluid)
    h3 = h1 + (h3is - h1) / eta
    c3  = c1              
    h3t = h3 + 0.5 * c3**2
    s3  = CP.PropsSI("S", "P", p3, "H", h3, fluid)
    T3  = CP.PropsSI("T", "P", p3, "H", h3, fluid)
    d3  = CP.PropsSI("D", "P", p3, "H", h3, fluid)
    p3t = CP.PropsSI("P", "H", h3t, "S", s3, fluid)
    T3t = CP.PropsSI("T", "H", h3t, "S", s3, fluid)
    d3t = CP.PropsSI("D", "H", h3t, "S", s3, fluid)

    a1 = CP.PropsSI("A", "P", p1, "T", T1, fluid)
    a2 = CP.PropsSI("A", "P", p2, "T", T2, fluid)
    a3 = CP.PropsSI("A", "P", p3, "T", T3, fluid)

    M1  = w1 / a1 
    M2R = w2 / a2
    M2S = c2 / a2
    M3  = c3 / a3

    mu1 = CP.PropsSI("V", "P", p1, "T", T1, fluid)
    mu2 = CP.PropsSI("V", "P", p2, "T", T2, fluid)
    mu3 = CP.PropsSI("V", "P", p3, "T", T3, fluid)

    TState1 = [p1, p1t, p1tr, T1, T1t, T1tr, d1, d1t, h1, h1t, h1tr, h1,   s1, c1, M1,  None, mu1, a1]
    TState2 = [p2, p2t, p2tr, T2, T2t, T2tr, d2, d2t, h2, h2t, h2tr, h2is, s2, c2, M2R, M2S,  mu2, a2]
    TState3 = [p3, p3t, None, T3, T3t, None, d3, d3t, h3, h3t, None, h3is, s3, c3, None, M3,  mu3, a3]

    return TState1, TState2, TState3


def PlotTS(state1, state2, state3):
    """Plot static temperature vs entropy (T-s diagram) with pressures in a side box."""
    # Extract values
    T_vals = [state1.T, state2.T, state3.T]
    s_vals = [state1.s, state2.s, state3.s]
    p_static = [state1.p/1e5, state2.p/1e5, state3.p/1e5]  # in bar
    p_total  = [state1.pt/1e5, state2.pt/1e5, state3.pt/1e5]  # in bar

    fig, ax = plt.subplots()
    ax.plot(s_vals, T_vals, 'o-', linewidth=1.5)
    for i, (si, Ti) in enumerate(zip(s_vals, T_vals), 1):
        ax.text(si, Ti, f'S{i}', fontsize=10, fontweight='bold',
                va='bottom', ha='right')

    # Prepare info box text
    info_lines = []
    for i, (ps, pt) in enumerate(zip(p_static, p_total), 1):
        info_lines.append(f'S{i}: Static={ps:.2f} bar, Total={pt:.2f} bar')
    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5))

    ax.set_xlabel('Entropy [J/kg·K]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('T-s Diagram with Pressures')
    ax.grid(True)
    plt.show()
    
def PlotHS(state1, state2, state3):
    """
    Plot an enthalpy-entropy (h-s) diagram for three states.
    """
    # Extract values
    h_values = [state1.ht, state1.h, state2.h, state3.h, state3.ht]
    s_values = [state1.s , state1.s, state2.s, state3.s, state3.s]
    
    # Plot curve
    plt.figure(figsize=(8, 6))
    plt.plot(s_values, h_values, marker='o', linestyle='-', color='b')
    plt.annotate(rf'State 1$_t$ (p$_t$={state1.pt / 1e5:.2f} bar)', xy=(s_values[0], h_values[0]), xytext=(s_values[0] + 5, h_values[0]))
    plt.annotate(f'State 1 (p={state1.p / 1e5:.2f} bar)', xy=(s_values[1], h_values[1]), xytext=(s_values[1] + 5, h_values[1]))
    plt.annotate(f'State 2 (p={state2.p / 1e5:.2f} bar)', xy=(s_values[2], h_values[2]), xytext=(s_values[2] + 5, h_values[2]))
    plt.annotate(f'State 3 (p={state3.p / 1e5:.2f} bar)', xy=(s_values[3], h_values[3]), xytext=(s_values[3] + 5, h_values[3]))      
    plt.annotate(f'State 3$_t$ (p={state3.pt / 1e5:.2f} bar)', xy=(s_values[4], h_values[4]), xytext=(s_values[4] + 5, h_values[4]))                      
    plt.xlabel('Entropy (s) [J/(kg·K)]', fontsize=12)
    plt.ylabel('Enthalpy (h) [J/kg]', fontsize=12)
    plt.xlim(min(s_values) - (max(s_values) - min(s_values)) * 0.1 , max(s_values) + (max(s_values) - min(s_values)) * 0.5)
    plt.ylim(min(h_values) - (max(h_values) - min(h_values)) * 0.2 , max(h_values) + (max(h_values) - min(h_values)) * 0.2)
    plt.grid()
    # plt.legend()
    plt.savefig('Thermodynamics Axial Compressor.png', dpi=300)
    plt.show()

def Radial(phim, psim, rm, Dm, D, u):
    
    D_ratio = Dm / D
    phi = phim *  D_ratio
    psi = psim * D_ratio**2
    r   = 1 - (1-rm) * D_ratio**2
    
    # Hub to tip diameter ratio or boss ratio
    M       = max(np.sqrt(1-rm), np.sqrt(psim), phim)
    bossmin = (2/M - 1)**-1
    urad    = u / (D/Dm)
    rad     = [phi, psi, r]
    
    return rad, urad, bossmin

def BladeLosses(
    TState1: Any,
    TState2: Any,
    BladeGeometry: Any,
    VelocityTriangle: Any,
    rotor: bool,
) -> float:
    
    mu1  = TState1.mu                 # inlet kinematic viscosity
    C      = BladeGeometry.C          # chord length
    hB     = BladeGeometry.hB         # blade height
    hB1    = BladeGeometry.hB1
    hB2    = BladeGeometry.hB2
    S      = BladeGeometry.S          # blade spacing
    sol    = BladeGeometry.sol        # solidity (s/C)
    tau    = BladeGeometry.tau        # tip clearance ratio (assumption) 0.004
    tmax   = BladeGeometry.tmax       # max blade thickness
    k      = BladeGeometry.k          # surface roughness
    cm     = VelocityTriangle.cm
    
    if rotor:
        Ma1, Ma2  = TState1.MR, TState2.MR
        v1, v2    = VelocityTriangle.w1, VelocityTriangle.w2
        a1, a2    = BladeGeometry.a1B, BladeGeometry.a2B      
    else:
        Ma1, Ma2  = TState1.MS, TState2.MS
        v1, v2    = VelocityTriangle.c2, VelocityTriangle.c1
        a1, a2    = BladeGeometry.a2, VelocityTriangle.a1

    # Ma1       = v1 / TState1.a
    # Ma2       = v2 / TState2.a

    # ---------- Profile Losses ----------

    # Constants were obtained experimentally by Koch and Smith
    K1, K2, K3, K4 = 0.2445, 0.4458, 0.7688, 0.6024
    
    # Assumptions
    tmaxC = 0.1
    Ma_x1 = cm / TState1.a
    
    A1 = hB1 * S
    A2 = hB2 * S
    
    Gamma = (np.tan(a1) - np.tan(a2)) * np.cos(a1) / sol
    
    A_throat = A1 - (1.0/3.0) * (A1 - A2)
    
    A_star_throat = (1.0 - K2 * sol * (tmaxC) /
                     (np.cos((a1+a2) / 2.0))) * (A_throat / A1)
    
    rhoth_rho1 = 1.0 - (Ma_x1**2)/(1.0 - Ma_x1**2) * (1.0 - A_star_throat 
                                                      - K1 * sol * Gamma * np.tan(a1) / np.cos(a1))
    
    DFeq = ((v1 / v2) * (1.0 + K3 * (tmax / C) + K4 * Gamma) 
            * np.sqrt((np.sin(a1) - K1 * sol * Gamma)**2 +
                      (np.cos(a1) / (A_star_throat * rhoth_rho1))**2))
    
    # Kinetic energy loss coefficient at trailing edge
    if DFeq <= 2:
        H0TE = (0.91 + 0.35 * DFeq) * (1 + 0.48 * (DFeq - 1)**4 + 0.21 * (DFeq - 1)**6)
    else:
        H0TE = 2.7209
    
    # Correcton factior of theta2_0_c for inlet Mach number
    n     = 2.853 + DFeq * (-0.97747 + 0.19477*DFeq)
    zetaM = 1.0 + (0.11757 - 0.16983*DFeq) * Ma1**n
    
    # Correction factor of theta2_0_c for the flow area contraction
    zetaH = 0.53 * hB1/hB2 + 0.47 ################ hB1/hB2
    
    # Critical Reynolds number and Re1
    Re_cr = 100.0 * C/k
    Re1   = v1 * C / mu1
    
    if Re1 <= Re_cr:
        # loss based on actual Re1
        if Re1 >= 2e5:
            zetaRe = (1e6 / Re1)**0.166
        else:
            zetaRe = 1.30626 * (2e5 / Re1)**0.5
    else:
        # loss based on critical Re_cr
        if Re_cr >= 2e5:
            zetaRe = (1e6 / Re_cr)**0.166
        else:
            zetaRe = 1.30626 * (2e5 / Re_cr)**0.5
                    
    # Correction factors of H0TE
    xiM = 1.0 + (1.0725 + DFeq * (-0.8671 + 0.18043 * DFeq)) * Ma1**1.8
    xiH = 1.0 + ((hB1 / hB2) - 1) * (0.0026 * DFeq**8 - 0.024)
    if Re1 < Re_cr:
        xiRe = (1e6 / Re1)**0.06 if Re1 >= 2e5 else 1.30626 * (2e5 / Re1)**0.5
    else:
        xiRe = (1e6 / Re_cr)**0.06 if Re_cr >= 2e5 else 1.30626 * (2e5 / Re_cr)**0.5
        
    # Loss Coefficient
    theta2_0_C = (2.644e-3 * DFeq - 1.519e-4 + (6.713e-3 / (2.60 - DFeq)))
    # Boundary layer momentum thickness at the blade outlet
    theta2_C = theta2_0_C * zetaM * zetaH * zetaRe
    
    # Boundary layer trailing edge shape factor
    HTE = H0TE * xiH * xiM * xiRe
    # Profile Loss Calculation
    Yp = 2 * theta2_C * (sol / np.cos(a2)) * (np.cos(a1) / np.cos(a2))**2 * (
        (2*HTE) / (3*HTE - 1)) * (1 - theta2_C * (sol*HTE / np.cos(a2)))**-3
        
    # ---------- Secondary Losses ----------
    
    # Mean flow angle
    tan_am = 0.5 * (np.tan(a1) + np.tan(a2))
    cos_am = 1.0 / np.sqrt(1.0 + tan_am**2)  ##################
    
    # Blade lift coefficient
    cL = 2/sol * cos_am * ((np.tan(a1) - np.tan(a2)))
    
    # Secondary Loss Calculation
    Ys = 0.018 * sol * (np.cos(a1)**2) / cos_am**3 * cL**2
    
    # ---------- End Wall Losses ----------
    
    Yew = 0.0146 * (C/hB) * (np.cos(a1) / np.cos(a2))**2
    
    # ---------- Shock Wave Losses ----------

    if Ma1 >= 1.0:
        Y_shock = 0.32 * Ma1**2 - 0.62 * Ma1 + 0.30
    else:
        Y_shock = 0.0
    
    # ---------- Tip Clearance Losses ----------

    # Assumptions
    KE, KG = 0.566, 0.943 # For fron or aft loaded blades
    # KE, KG = 0.5, 1.0 # For mid loaded blades for l/C=0.5

    # Tip and Gap losses
    Ytip = 1.4 * KE * sol * (tau / hB) * np.cos(a1)**2 / cos_am**3 * cL**1.5
    Ygap = 0.0049 * KG * sol * C/hB * np.sqrt(cL) / cos_am
    
    YTC = Ytip + Ygap
    
    # ----------------- Total Blade Loss -----------------

    #     # --- NEW RETURN ---
    # loss_components = {
    #     'Yp': Yp,
    #     'Ys': Ys,
    #     'Yew': Yew,
    #     'Y_shock': Y_shock,
    #     'YTC': YTC
    # }
    # Y = Yp + Ys + Yew + Y_shock + YTC  
    # return loss_components
    Y = Yp + Ys + Yew + Y_shock + YTC
    
    return Y

# def Efficiency(VelocityTriangle, TState1, TState2, TState3, YR, YS, fluid):
    
#     # Unpack velocities and states
#     c2 = VelocityTriangle.c2
#     w1 = VelocityTriangle.w1
    
#     h1, h1t, h1tr = TState1.h, TState1.ht, TState1.htr
#     p1, p1t, p1tr = TState1.p, TState1.pt, TState1.ptr
#     T1, T1t = TState1.T, TState1.Tt
#     h2is = TState2.his
#     h3is = TState3.his
    
#     s1 = CP.PropsSI("S", "P", p1, "T", T1, fluid)

#     # Total relative pressure downstream of the rotor
#     p2tr = p1tr - YR * (p1tr - p1)
    
#     # Total relative enthalpy downstream of the rotor is constant in an axial stage
#     h2tr = h1tr     
    
#     # Thermodynamic quantities downstream of the rotor
#     s2 = CP.PropsSI("S", "P", p2tr, "H", h2tr, fluid)
#     h2 = CP.PropsSI("H", "P", TState2.p , "S", s2, fluid) # Use state 2 static pressure
    
#     # For stator efficiency
#     h2t = h2 + 0.5 * c2**2 
#     p2t = CP.PropsSI("P", "S", s2, "H", h2t, fluid)
    
#     # Total pressure downstream of the stator
#     p3t = p2t - YS * (p2t - p2t) # Note: Stator loss is based on dynamic pressure at stator inlet (station 2)
#     h3t = h2t
    
#     # Thermodynamic quantities downstream of the stator
#     s3  = CP.PropsSI("S", "P", p3t, "H", h3t, fluid)
#     h3  = CP.PropsSI("H", "S", s3 , "P", TState3.p, fluid) # Use state 3 static pressure
#     h3tis = CP.PropsSI("H", "P", p3t, "S", s1, fluid)

#     # Calculate final efficiencies
#     etaR  = (h2is - h1)   / (h2 - h1) if (h2-h1) != 0 else 0
#     eta   = (h3is - h1)   / (h3 - h1) if (h3-h1) != 0 else 0
#     etaTT = (h3tis - h1t) / (h3t - h1t) if (h3t-h1t) != 0 else 0
#     etaTS = (h3is - h1t)  / (h3t - h1t) if (h3t-h1t) != 0 else 0
    
#     return eta, etaR, etaTT, etaTS

def Efficiency(VelocityTriangle, TState1, TState2, TState3, YR, YS, fluid):
    
    # Unpack velocities
    c2   = VelocityTriangle.c2
    w1   = VelocityTriangle.w1
    
    # State 1
    h1, h1t  = TState1.h, TState1.ht
    p1, p1t  = TState1.p, TState1.pt
    T1, T1t  = TState1.T, TState1.Tt
    h1tr = h1 + 0.5*w1**2
    s1 = CP.PropsSI("S", "P", p1, "T", T1, fluid)
    
    # Total relative pressure
    p1tr = CP.PropsSI("P", "H", h1tr, "S", s1, fluid)
    
    # State 2
    p2   = TState2.p
    h2   = TState2.h
    h2t  = TState2.ht
    h2is = TState2.his
    
    # Total relative pressure downstream of the rotor
    p2tr = p1tr - YR * (p1tr - p1)
    # Total relative enthalpy downstream of the rotor
    h2tr = h1tr     
    
    # State 3
    p3   = TState3.p
    h3   = TState3.h
    # h3t  = TState3.ht
    h3is = TState3.his
    
    # Thermodynamic quantities downstream of the rotor
    s2 = CP.PropsSI("S", "P", p2tr, "H", h2tr, fluid)
    h2 = CP.PropsSI("H", "P", p2  , "S", s2  , fluid)
    
    # For stator efficiency
    h2t = h2 + 0.5 * c2**2 
    p2t = CP.PropsSI("P", "S", s2, "H", h2t, fluid)
    p3t = p2t - YS * (p2t - p2)
    h3t = h2t
    h3tis = CP.PropsSI("H", "P", p3t, "S", s1, fluid)
    s3  = CP.PropsSI("S", "P", p3t, "H", h3t, fluid)
    h3  = CP.PropsSI("H", "S", s3 , "P", p3 , fluid)
    
    # Efficiencies
    etaR  = (h2is - h1)   / (h2 - h1) if (h2-h1) != 0 else 0
    eta   = (h3is - h1)   / (h3 - h1) if (h3-h1) != 0 else 0
    etaTT = (h3tis - h1t) / (h3t - h1t) if (h3t-h1t) != 0 else 0
    etaTS = (h3is - h1t)  / (h3t - h1t) if (h3t-h1t) != 0 else 0
    # etaR  = (h2is - h1)   / (h2 - h1)
    # eta   = (h3is - h1)   / (h3 - h1)
    # etaTT = (h3tis - h1t) / (h3t - h1t)
    # etaTS = (h3is - h1t)  / (h3t - h1t)
    
    # return etaR, etaTT, eta, etaTS
    return eta, etaR, etaTT, etaTS
