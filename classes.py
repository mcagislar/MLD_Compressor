import numpy as np
import matplotlib.pyplot as plt

class VelocityTriangle:
    def __init__(self, psi, phi, r, u, angles, v_abs, v_tg):
        
        self.psi, self.phi, self.r,   self.u            = psi, phi, r, u
        self.a1,  self.a2,  self.b1,  self.b2           = angles
        self.cm,  self.c1,  self.c2,  self.w1, self.w2  = v_abs
        self.c1u, self.c2u, self.w1u, self.w2u          = v_tg
        self.c3 = self.c1
        
    def _plotVT(self, ax, x_start, y_start, x_end, y_end, color, label):
        ax.quiver(x_start, y_start, x_end, y_end, angles='xy', scale_units='xy', scale=1, color=color, label=label)
        
    def plotVT(self, ax=None, figsize=(9, 7), save_path='velocity_triangles.pdf'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        colors = {
            'meridional'        : 'black',
            'absolute_stator'   : 'blue',
            'relative_stator'   : '#66b3ff',
            'absolute_rotor'    : 'red',
            'relative_rotor'    : '#ff6666',
            'blade_velocity1'   : 'green',
            'blade_velocity2'   : 'purple'
        }
        
        # self._plotVT(ax, 0, 0, 0, self.cm, colors['meridional'], r'Meridional Velocity $C_m$')
        self._plotVT(ax, 0, 0, self.c1u, self.cm, colors['absolute_rotor'], r'Absolute Velocity $C_1$ (Rotor Inlet)')
        self._plotVT(ax, 0, 0, self.w1u, self.cm, colors['relative_rotor'], r'Relative Velocity $W_1$ (Rotor Inlet)')
        self._plotVT(ax, 0, 0, self.c2u, self.cm, colors['absolute_stator'], r'Absolute Velocity $C_2$ (Rotor Exit)')
        self._plotVT(ax, 0, 0, -self.w2u, self.cm, colors['relative_stator'], r'Relative Velocity $W_2$ (Rotor Exit)')
        self._plotVT(ax, -self.w2u, self.cm, self.u, 0, colors['blade_velocity2'], r'Blade Velocity $U$')
        self._plotVT(ax, 0, self.cm, self.u, 0, colors['blade_velocity1'], r'Blade Velocity $U$')
        
        # # self._plotVT(ax, 0, 0, 0, self.cm, colors['meridional'], r'Meridional Velocity $C_m$')
        # self._plotVT(ax, 0, 0, self.c1u, self.cm, colors['absolute_rotor'], r'Absolute Velocity $C_1$ (Rotor Inlet)')
        # self._plotVT(ax, 0, 0, self.w1u, self.cm, colors['relative_rotor'], r'Relative Velocity $W_1$ (Rotor Inlet)')
        # self._plotVT(ax, -self.u + self.cm, 0, self.c2u, self.cm, colors['absolute_stator'], r'Absolute Velocity $C_2$ (Rotor Exit)')
        # self._plotVT(ax, -self.u + self.cm, 0, -self.w2u, self.cm, colors['relative_stator'], r'Relative Velocity $W_2$ (Rotor Exit)')
        # # self._plotVT(ax, -self.c2u, self.cm, self.u, 0, colors['blade_velocity2'], r'Blade Velocity $U$')
        # self._plotVT(ax, 0, self.cm, self.u, 0, colors['blade_velocity1'], r'Blade Velocity $U$')

        # ax.set_aspect('equal')
        max_x = max(self.c1u, self.c2u, self.w1u, self.w2u, self.u)
        ax.set_xlim(-1.4*max_x, 1.6*max_x)
        ax.set_ylim(-1.3*self.cm, 1.5*self.cm)
        ax.set_xlabel("Velocity [m/s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.axhline(0, color='k', linewidth=0.7, linestyle='--')
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
        ax.legend(loc='best', fontsize=7, frameon=True, facecolor='white', edgecolor='gray')
        ax.invert_yaxis()
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        plt.show()

class TState:
    def __init__(self, state):
         self.p, self.pt, self.ptr = state[0:3]
         self.T, self.Tt, self.Ttr = state[3:6]
         self.d, self.dt = state[6:8]
         self.h, self.ht, self.htr, self.his = state[8:12]
         self.s, self.c = state[12:14]
         self.MR, self.MS = state[14:16]
         self.mu, self.a = state[16:18]

class StageGeometry:
    def __init__(self, VelocityTriangle, TState1, TState2, TState3, m, Dm):
        
        cm = VelocityTriangle.cm

        self.V1 = m / TState1.d
        self.V2 = m / TState2.d
        self.V3 = m / TState3.d

        self.A1 = self.V1 / cm
        self.A2 = self.V2 / cm
        self.A3 = self.V3 / cm

        self.hB1 = self.A1 / (np.pi * Dm)
        self.hB2 = self.A2 / (np.pi * Dm)
        self.hB3 = self.A3 / (np.pi * Dm)

        self.boss1 = (1 - self.hB1 / Dm) / (1 + self.hB1 / Dm)
        self.boss2 = (1 - self.hB2 / Dm) / (1 + self.hB2 / Dm)
        self.boss3 = (1 - self.hB3 / Dm) / (1 + self.hB3 / Dm)

        self.Dt1 = 2 * Dm / (1 + self.boss1)
        self.Dh1 = 2 * Dm * self.boss1 / (1 + self.boss1)
        self.Dt2 = 2 * Dm / (1 + self.boss2)
        self.Dh2 = 2 * Dm * self.boss2 / (1 + self.boss2)
        self.Dt3 = 2 * Dm / (1 + self.boss3)
        self.Dh3 = 2 * Dm * self.boss3 / (1 + self.boss3)   

class BladeGeometry:
    def __init__(self, vt: VelocityTriangle, sg: StageGeometry,
                 TState1, TState2, TState3, Dm, rotor):
        self.cm = vt.cm
        self.c2, self.c3 = vt.c2, vt.c1
        self.w1, self.w2 = vt.w1, vt.w2
       
        mu1, mu2, mu3    = TState1.mu, TState2.mu, TState3.mu
        rho1, rho2, rho3 = TState1.d , TState2.d , TState3.d
        
        DF, tmaxC, lC, tmaxhB    = 0.45, 0.05, 0.70, 0.03
        
        self.hB1  = sg.A1 / (np.pi * Dm)
        self.hB2  = sg.A2 / (np.pi * Dm)     
        self.hB3  = sg.A3 / (np.pi * Dm)
        
        self.tau = 0.0005 if rotor else 0.0
        self.tTE    = 0.0005
        self.tau_hB = 0.6
        self.k = 50e-6

        self.Kt    = (10 * tmaxC)**(0.28 / (0.1 + tmaxC)**0.3)
        self.Kti   = 6.25 * tmaxC + 37.5 * tmaxC**2

        if rotor:
            self.a1, self.a2 = vt.b1, vt.b2
            print(np.rad2deg(self.a1), np.rad2deg(self.a2))
            self.Dh   = 0.5 * (sg.Dh1 + sg.Dh2)
            self.Dt   = 0.5 * (sg.Dt1 + sg.Dt2)
            self.boss = 0.5 * (sg.boss1 + sg.boss2)
            self.hB   = 0.5 * (sg.hB1 + sg.hB2)
            self.stagg  = 0.5 * (self.a1 + self.a2)
            # self.AR_opt = (0.316 * tmaxC**(-0.416))**(1/0.584)
            self.AR_opt = 0.316 * tmaxhB**(-0.416)
            self.AR_min = 0.8 * self.AR_opt
            self.AR_max = 1.2 * self.AR_opt
            self.C      = self.hB / self.AR_opt                     
            self.tmax = self.C * tmaxC
            self.ratio = self.tmax / self.hB
            self.solden = 2 * (DF - 1 + (np.cos(self.a1) / np.cos(self.a2)))
            self.sol    = (np.cos(self.a1) / self.solden) * (vt.psi/vt.phi)
            self.Ca     = self.C * np.cos(self.stagg)
            self.S      = self.C / self.sol
            CR          = self.C
            CS          = self.S
            a1_deg = np.rad2deg(self.a1)
            a2_deg = np.rad2deg(self.a2)
            deltA = abs(a1_deg - a2_deg)
            if not (10.0 <= a1_deg <= 65.0):
                raise ValueError(f"β1 ({a1_deg:.1f}°) must be between 10° and 65°")
            if not (0.0 <= a2_deg <= 55.0):
                raise ValueError(f"β2 ({a2_deg:.1f}°) must be between 0° and 55°")
            if not (10.0 <= deltA <= 60.0):
                raise ValueError(f"Angle deflection ({deltA:.1f}°) must be between 10° and 60°")
            self.A = -0.0197 + 0.042231 * deltA
            self.B = np.exp(-13.427 + deltA * (0.33303 - 0.002368 * deltA))
            n = 2.8592 - 0.04677 * deltA
            self.solopt = self.A + self.B * (self.a2)**n
            
            self.delta0st = (0.01 * self.sol * self.a1 + (0.74 * self.sol**1.9 + 3 * self.sol)
                          * (self.a1 / 90) ** (1.67 + 1.09 * self.sol))
            # Initial guesses in DEGREES
            theta_rad = 0.0872665  # Blade camber angle
            deltaw_rad = 0.0349066 # Deviation angle
            inc = 0.0    # Incidence angle
            
            tolerance, max_iter = 1e-4, 100
            for _ in range(max_iter):
                prev_theta_rad = theta_rad
                theta_rad = (self.a1 - self.a2) + deltaw_rad - inc
                
                eC = (np.sqrt(1 + (4 * np.tan(theta_rad))**2 * (lC - lC**2 - 3/16)) - 1) / (4 * np.tan(theta_rad))
            
                chi1 = np.arctan(eC / (lC - 0.25))
                chi2 = np.arctan(eC / (0.75 - lC))
                # self.stagg = self.a1 - (3.6 * self.Kt + 0.3532 * theta_rad * (lC)**0.25) * (
                #                           self.sol)**(0.65 - 0.02 * theta_rad)
                self.a1B = self.stagg + chi1
                self.a2B = self.stagg - chi2
                inc = self.a1 - self.a1B
            
                Ksh      = 1.0 
                Kt_prime = self.Kti
                dwterm1  = (Ksh * Kt_prime - 1) * self.delta0st
                dwterm2_num = 0.92 * (lC)**2 + 0.002 * self.a2B
                dwterm2_den = 1.0 - (0.002 * theta_rad) / np.sqrt(self.sol)
                dwterm2 = (dwterm2_num / dwterm2_den) * (theta_rad / np.sqrt(self.sol))
                deltaw_rad = dwterm1 + dwterm2
            
                if abs(theta_rad - prev_theta_rad) < tolerance:
                    break
            
            # Store final values
            self.theta = theta_rad
            self.deltaw = deltaw_rad
            self.inc = inc
            
        else:
            self.a1, self.a2 = vt.a2, vt.a1
            print(np.rad2deg(self.a1), np.rad2deg(self.a2))
            self.Dh   = 0.5 * (sg.Dh2 + sg.Dh3)
            self.Dt   = 0.5 * (sg.Dt2 + sg.Dt3)
            self.boss = 0.5 * (sg.boss2 + sg.boss3)
            self.hB   = 0.5 * (sg.hB2 + sg.hB3)
            self.stagg  = 0.5 * (self.a1 + self.a2)
            # self.AR_opt = (0.316 * tmaxC**(-0.416))**(1/0.584)
            self.AR_opt = 0.316 * tmaxhB**(-0.416)
            self.AR_min = 0.8 * self.AR_opt
            self.AR_max = 1.2 * self.AR_opt
            self.C      = self.hB / self.AR_opt                     
            self.tmax = self.C * tmaxC
            self.ratio = self.tmax / self.hB
            self.solden = 2 * (DF - 1 + (np.cos(self.a1) / np.cos(self.a2)))
            self.sol    = (np.cos(self.a1) / self.solden) * (vt.psi/vt.phi)
            self.Ca     = self.C * np.cos(self.stagg)
            self.S      = self.C / self.sol
            CR          = self.C
            CS          = self.S
            a1_deg = np.rad2deg(self.a1)
            a2_deg = np.rad2deg(self.a2)
            deltA = abs(a1_deg - a2_deg)
            if not (10.0 <= a1_deg <= 65.0):
                raise ValueError(f"α1 ({a1_deg:.1f}°) must be between 10° and 65°")
            if not (0.0 <= a2_deg <= 55.0):
                raise ValueError(f"α2 ({a2_deg:.1f}°) must be between 0° and 55°")
            if not (10.0 <= deltA <= 60.0):
                raise ValueError(f"Angle deflection ({deltA:.1f}°) must be between 10° and 60°")
            self.A = -0.0197 + 0.042231 * deltA
            self.B = np.exp(-13.427 + deltA * (0.33303 - 0.002368 * deltA))
            n = 2.8592 - 0.04677 * deltA
            self.solopt = self.A + self.B * (self.a2)**n
            
            self.delta0st = (0.01 * self.sol * self.a1 + (0.74 * self.sol**1.9 + 3 * self.sol)
                          * (self.a1 / 90) ** (1.67 + 1.09 * self.sol))
            
            # Initial guesses in Radians
            theta_rad = 0.0872665   # Blade camber angle
            deltaw_rad = 0.0349066  # Deviation angle
            inc = 0.0               # Incidence angle
            
            tolerance, max_iter = 1e-4, 100
            for _ in range(max_iter):
                prev_theta_rad = theta_rad
                theta_rad = self.a1 - self.a2 + deltaw_rad - inc
                
                eC = (np.sqrt(1 + (4 * np.tan(theta_rad))**2 * (lC - lC**2 - 3/16)) - 1) / (4 * np.tan(theta_rad))
            
                chi1 = np.arctan(eC / (lC - 0.25))
                chi2 = np.arctan(eC / (0.75 - lC))
                # self.stagg = self.a1 - (3.6 * self.Kt + 0.3532 * theta_rad * (lC)**0.25) * (
                #                           self.sol)**(0.65 - 0.02 * theta_rad)
                self.a1B = self.stagg + chi1
                self.a2B = self.stagg - chi2
                inc = self.a1 - self.a1B
            
                Ksh      = 1.0 
                Kt_prime = self.Kti
                dwterm1  = (Ksh * Kt_prime - 1) * self.delta0st
                dwterm2_num = 0.92 * (lC)**2 + 0.002 * self.a2B
                dwterm2_den = 1.0 - (0.002 * theta_rad) / np.sqrt(self.sol)
                dwterm2 = (dwterm2_num / dwterm2_den) * (theta_rad / np.sqrt(self.sol))
                deltaw_rad = dwterm1 + dwterm2
            
                if abs(theta_rad - prev_theta_rad) < tolerance:
                    break
            
            # Store final values
            self.theta = theta_rad
            self.deltaw = deltaw_rad
            self.inc = inc
                
        self.NB     = np.round(np.pi * Dm / self.S)
        self.O      = self.S * np.cos(self.a1B)
        
        self.Re1    = rho1 * self.w1 * CR / mu1 
        self.Re2_R  = rho2 * self.w2 * CR / mu2
        self.Re2_S  = rho2 * self.c2 * CS / mu2
        self.Re3    = rho3 * self.c3 * CS / mu3
        
        self.ks_adm_R = 100 * CR / self.Re1
        self.ks_adm_S = 100 * CS / self.Re2_S
                
        # if rotor:
        #     self.a1, self.a2, self.b2 = vt.a1, vt.b1, vt.b2
        #     if self.a1<0: self.a1 = 0
        #     print(np.rad2deg(self.a1), np.rad2deg(self.a2))
        #     self.Dh   = 0.5 * (sg.Dh1 + sg.Dh2)
        #     self.Dt   = 0.5 * (sg.Dt1 + sg.Dt2)
        #     self.boss = 0.5 * (sg.boss1 + sg.boss2)
        #     self.hB   = 0.5 * (sg.hB1 + sg.hB2)
        #     self.stagg  = 0.5 * (self.a2 + self.b2)
        #     self.AR_opt = (0.316 * tmaxC**(-0.416))**(1/0.584)
        #     self.AR_min = 0.8 * self.AR_opt
        #     self.AR_max = 1.2 * self.AR_opt
        #     self.C      = self.hB / self.AR_opt                     
        #     self.tmax = self.C * tmaxC
        #     self.ratio = self.tmax / self.hB
        #     self.solden = 2 * (DF - 1 + (np.cos(self.a2) / np.cos(self.b2)))
        #     self.sol    = (np.cos(self.a2) / self.solden) * (vt.psi/vt.phi)
        #     self.Ca     = self.C * np.cos(np.rad2deg(self.stagg))
        #     self.S      = self.C / self.sol
        #     deltA = np.rad2deg(self.a2) - np.rad2deg(self.b2)
        #     if not (10.0 <= np.rad2deg(self.a2) <= 65.0):
        #         raise ValueError("β1 must be between 10° and 65°")
        #     if not (0.0 <= np.rad2deg(self.b2) <= 55.0):
        #         raise ValueError("β2 must be between 0° and 55°")
        #     if not (10.0 <= deltA <= 60.0):
        #         raise ValueError("β1 - β2 must be between 10° and 60°")
        #     self.A      = -0.0197 + 0.042231*deltA
        #     self.B      = np.exp(-13.427 + deltA * (0.33303 - 0.002368*deltA))
        #     n           = 2.8592 - 0.04677*deltA
        #     self.solopt = self.A + self.B * (np.rad2deg(self.b2) ** n)

        # else:
        #     self.a1, self.a2, self.b3 = vt.a2, vt.b2, 0.0
        #     print(np.rad2deg(self.a1), np.rad2deg(self.a2))
        #     self.Dh   = 0.5 * (sg.Dh2 + sg.Dh3)
        #     self.Dt   = 0.5 * (sg.Dt2 + sg.Dt3)
        #     self.boss = 0.5 * (sg.boss2 + sg.boss3)
        #     self.hB   = 0.5 * (sg.hB2 + sg.hB3)
        #     self.stagg  = 0.5 * (self.a2 + self.b3)
        #     self.AR_opt = (0.316 * tmaxC**(-0.416))**(1/0.584)
        #     self.AR_min = 0.8 * self.AR_opt
        #     self.AR_max = 1.2 * self.AR_opt
        #     self.C      = self.hB / self.AR_opt                     
        #     self.tmax = self.C * tmaxC
        #     self.ratio = self.tmax / self.hB
        #     self.solden = 2 * (DF - 1 + (np.cos(self.a2) / np.cos(self.b3)))
        #     self.sol    = (np.cos(self.a2) / self.solden) * (vt.psi/vt.phi)
        #     self.Ca     = self.C * np.cos(np.rad2deg(self.stagg))
        #     self.S      = self.C / self.sol
        #     deltA = np.rad2deg(self.a2) - np.rad2deg(self.b3)
        #     if not (10.0 <= np.rad2deg(self.a2) <= 65.0):
        #         raise ValueError("β1 must be between 10° and 65°")
        #     if not (0.0 <= np.rad2deg(self.b3) <= 55.0):
        #         raise ValueError("β2 must be between 0° and 55°")
        #     if not (10.0 <= deltA <= 60.0):
        #         raise ValueError("β1 - β2 must be between 10° and 60°")
        #     self.A      = -0.0197 + 0.042231*deltA
        #     self.B      = np.exp(-13.427 + deltA * (0.33303 - 0.002368*deltA))
        #     n           = 2.8592 - 0.04677*deltA
        #     self.solopt = self.A + self.B * (np.rad2deg(self.b3) ** n)

        #     # self.NB     = np.round(np.pi * Dm / self.S)
        #     # self.O      = self.S * np.cos(np.deg2rad(self.a1B))
        #     # self.tau = 0.0005 if rotor else 0.0
        #     # self.tTE    = 0.0005
        #     # self.tau_hB = 0.01
        #     # self.O      = self.S * np.cos(np.deg2rad(self.a1B))
        


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        