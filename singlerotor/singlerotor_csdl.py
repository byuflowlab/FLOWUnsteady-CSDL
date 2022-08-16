from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

from julia import FLOWUnsteady
from julia import FLOWVLM

import math

from csdl_om import Simulator
from csdl import Model
import csdl
import numpy as np

extdrive_path = "/home/christian/Documents/FlowU_CSDL/"
def singlerotor(xfoil       = True,             # Whether to run XFOIL
                VehicleType = FLOWUnsteady.VLMVehicle,   # Vehicle type
                J           = 0.0,              # Advance ratio
                DVinf       = [1.0, 0, 0],      # Freestream direction
                nrevs       = 6,                # Number of revolutions
                nsteps_per_rev = 72,            # Time steps per revolution
                shed_unsteady = True,
                lmd     = 2.125,
                n           = 10,
                overwrite_overwrite_sigma = None,
                # OUTPUT OPTIONS
                save_path   = None,
                run_name    = "singlerotor",
                prompt      = True,
                verbose     = True,
                v_lvl       = 0,
                rotor_file = "DJI-II.csv"):         # Rotor geometry

    # TODO: Wake removal ?

    # ------------ PARAMETERS --------------------------------------------------

    # Rotor geometry
    data_path = FLOWUnsteady.def_data_path       # Path to rotor database
    pitch = 0.0                         # (deg) collective pitch of blades
    # n = 50                              # Number of blade elements
    # n = 10
    CW = False                          # Clock-wise rotation
    # xfoil = False                     # Whether to run XFOIL

    # Read radius of this rotor and number of blades
    a = FLOWUnsteady.read_rotor(rotor_file, data_path=data_path)
    R = a[0]
    B = a[2]


    # Simulation parameters
    RPM = 81*60;                         # RPM
    # J = 0.00001                       # Advance ratio Vinf/(nD)
    rho = 1.225                         # (kg/m^3) air density
    mu = 1.81e-5                        # (kg/ms) air dynamic viscosity
    ReD = 2*math.pi*RPM/60*R * rho/mu * 2*R  # Diameter-based Reynolds number
    magVinf = J*RPM/60*(2*R)
    Main.magVinf = magVinf;
    Main.DVinf = DVinf;
    Vinf = Main.eval("Vinf(X,t) = magVinf*DVinf")          # (m/s) freestream velocity
    # Solver parameters
    # nrevs = 6                         # Number of revolutions in simulation
    # nsteps_per_rev = 72                 # Time steps per revolution
    p_per_step = 2;                      # Sheds per time step
    ttot = nrevs/(RPM/60);              # (s) total simulation time
    nsteps = nrevs*nsteps_per_rev       # Number of time steps
    # lambda = 2.125                      # Core overlap
    if(overwrite_overwrite_sigma != None):
        overwrite_sigma = overwrite_overwrite_sigma;
    else:
        overwrite_sigma = lmd * 2*math.pi*R/(nsteps_per_rev*p_per_step);

    surf_sigma = R/10                   # Smoothing radius of lifting surface
    vlm_sigma = surf_sigma              # Smoothing radius of VLM
    # shed_unsteady = True                # Shed particles from unsteady loading

    max_particles = ((2*n+1)*B)*nrevs*nsteps_per_rev*p_per_step # Max particles for memory pre-allocation
    plot_disc = True                    # Plot blade discretization for debugging


    # ------------ SIMULATION SETUP --------------------------------------------
    # Generate rotor
    rotor = FLOWUnsteady.generate_rotor(rotor_file, pitch=pitch,
                                            n=n, CW=CW, ReD=ReD,
                                            verbose=verbose, xfoil=xfoil,
                                            data_path=data_path,
                                            plot_disc=plot_disc)
    # ----- VEHICLE DEFINITION
    # System of all FLOWVLM objects
    system = FLOWVLM.WingSystem()
    FLOWVLM.addwing(system, run_name, rotor)

    # Systems of rotors
    rotors = [rotor]; # Defining this rotor as its own system
    rotor_systems = (rotors,)

    # Wake-shedding system (doesn't include the rotor if quasi-steady vehicle)
    wake_system = FLOWVLM.WingSystem()

    if VehicleType != FLOWUnsteady.QVLMVehicle:
        FLOWVLM.addwing(wake_system, run_name, rotor)
    else:
        # Mute colinear warnings. This is needed since the quasi-steady solver
        #   will probe induced velocities at the lifting line of the blade
        FLOWUnsteady.FLOWVLM.VLMSolver._mute_warning(True)

    # FVS's Vehicle object
    vehicle = VehicleType(   system,
                                rotor_systems=rotor_systems,
                                wake_system=wake_system
                             )
    RPM_fun = Main.eval("RPM_fun(t) = 1.0")               # RPM (normalized by reference RPM) as a
                                    # function of normalized time

    angle = ()                      # Angle of each tilting system (none in this case)
    sysRPM = (RPM_fun, )              # RPM of each rotor system
    Vvehicle = Main.eval("Vvehicle(t) = zeros(3)")         # Translational velocity of vehicle over Vcruise
    anglevehicle = Main.eval("anglevehicle(t) = zeros(3)")      # (deg) angle of the vehicle

    # FVS's Maneuver object
    maneuver = FLOWUnsteady.KinematicManeuver(angle, sysRPM, Vvehicle, anglevehicle)

    # Plot maneuver path and controls
    FLOWUnsteady.plot_maneuver(maneuver, vis_nsteps=nsteps)


    # ----- SIMULATION DEFINITION
    RPMref = RPM
    Vref = 0.0
    simulation = FLOWUnsteady.Simulation(vehicle, maneuver, Vref, RPMref, ttot)

#    monitor = generate_monitor(J, rho, RPM, nsteps, save_path=save_path,
#                                                            run_name=run_name)
    def monitor():
        return False;
    # ------------ RUN SIMULATION ----------------------------------------------
    pfield = FLOWUnsteady.run_simulation(simulation, nsteps,
                                      # SIMULATION OPTIONS
                                      Vinf=Vinf,
                                      # SOLVERS OPTIONS
                                      p_per_step=p_per_step,
                                      overwrite_sigma=overwrite_sigma,
                                      vlm_sigma=vlm_sigma,
                                      surf_sigma=surf_sigma,
                                      max_particles=max_particles,
                                      shed_unsteady=shed_unsteady,
                                      # OUTPUT OPTIONS
                                      save_path=save_path,
                                      run_name=run_name,
                                      prompt=prompt,
                                      verbose=verbose, v_lvl=v_lvl)
    return pfield

class SingleRotor_Hover(Model):
    def define(self):
        J = self.create_input('J', val=0.00)
        J1 = J;
        angle = 0.0
        p = singlerotor(xfoil=True, VehicleType=FLOWUnsteady.VLMVehicle, J=0.00,
        DVinf=[math.cos(math.pi/180*angle), math.sin((math.pi)/180*angle), 0],save_path=extdrive_path + "singlerotor_csdl/", prompt=True)
        self.register_output('PField', p)

sim = Simulator(SingleRotor_Hover())
sim.run()
print('pfield', sim['PField'].shape)
