#=##############################################################################
# DESCRIPTION
#Test rotor model simulating an isolated 9.4in rotor in hover. The rotor and
#configuration matches the 3D-printed propeller described in Ning, Z.,
#*Experimental investigations on the aerodynamic and aeroacoustic characteristics
#of small UAS propellers*, Sec. 5.2. This rotor roughly resembles a DJI Phantom
#II rotor.
#
# AUTHORSHIP
#  * Author    : Eduardo J. Alvarez
#  * Email     : Edo.AlvarezR@gmail.com
#  * Created   : Dec 2019
#  * License   : MIT
#=###############################################################################

# ------------ MODULES ---------------------------------------------------------
# Load simulation engine
from julia.api import Julia
jl = Julia(compiled_modules=False)

from julia import Main

from julia import FLOWUnsteady
from julia import FLOWVLM

import math

# ------------ GLOBAL VARIABLES ------------------------------------------------
# Default path where to save data
extdrive_path = "/home/christian/Documents/FlowU_CSDL/"
# extdrive_path = "temps/"

# ------------ DRIVERS ---------------------------------------------------------
def run_singlerotor_hover(xfoil=True, prompt=True):

    J = 0.00                # Advance ratio Vinf/(nD)
    angle = 0.0             # (deg) angle of freestream (0 == climb, 90==forward flight)

    p = singlerotor(xfoil=xfoil,
                VehicleType=FLOWUnsteady.VLMVehicle,
                J=J,
                DVinf=[math.cos(math.pi/180*angle), math.sin((math.pi)/180*angle), 0],
                save_path=extdrive_path + "singlerotor_hover_test00/",
                prompt=prompt)
    return p;

def run_singlerotor_forwardflight(xfoil=True, prompt=True):

    J = 0.15                # Advance ratio Vinf/(nD)
    angle = 60.0            # (deg) angle of freestream (0 == climb, ~90==forward flight)

    p =  singlerotor( xfoil=xfoil,
                VehicleType=FLOWUnsteady.VLMVehicle,
                J=J,
                DVinf=[math.cos(math.pi/180*angle), math.sin((math.pi)/180*angle), 0],
                nrevs=2,
                nsteps_per_rev=120,
                save_path=extdrive_path + "singlerotor_fflight_test01/",
                prompt=prompt);
    return p;


# ------------------------------------------------------------------------------

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

#"""
#Generate monitor for rotor performance parameters
#"""
#def generate_monitor(J, rho, RPM, nsteps, save_path=None,
#                            run_name="singlerotor",
#                            figname="monitor", disp_conv=True,
#                            nsteps_savefig=10):
#
#    fcalls = 0                  # Number of function calls
#
#    colors="rgbcmy"^100
#    stls = "o^*.px"^100
#
#    # Name of convergence file
#    if (save_path!=None):
#        fname = joinpath(save_path, run_name*"_convergence.csv")
#
#    # Function for run_vpm! to call on each iteration
#    def extra_runtime_function(sim,
#                                    PFIELD,
#                                    T, DT):
#
#        rotors = vcat(sim.vehicle.rotor_systems...)
#        angle = T*360*RPM/60
#
#
#        # Call figure
#        if disp_conv; fig = figure(figname, figsize=(7*3,5*2)); end;
#
#        if fcalls==0
#            # Format subplots
#            if disp_conv
#                subplot(231)
#                title("Circulation Distribution")
#                xlabel("Element index")
#                ylabel(L"Circulation $\Gamma$ (m$^2$/s)")
#                grid(True, color="0.8", linestyle="--")
#                subplot(232)
#                title("Plane-of-rotation Normal Force")
#                xlabel("Element index")
#                ylabel(L"Normal Force $N_p$ (N)")
#                grid(True, color="0.8", linestyle="--")
#                subplot(233)
#                title("Plane-of-rotation Tangential Force")
#                xlabel("Element index")
#                ylabel(L"Tangential Force $T_p$ (N)")
#                grid(True, color="0.8", linestyle="--")
#                subplot(234)
#                xlabel(L"Age $\psi$ ($^\circ$)")
#                ylabel(L"Thrust $C_T$")
#                grid(True, color="0.8", linestyle="--")
#                subplot(235)
#                xlabel(L"Age $\psi$ ($^\circ$)")
#                ylabel(L"Torque $C_Q$")
#                grid(True, color="0.8", linestyle="--")
#                subplot(236)
#                xlabel(L"Age $\psi$ ($^\circ$)")
#                ylabel(L"Propulsive efficiency $\eta$")
#                grid(True, color="0.8", linestyle="--")
#            end
#
#            # Convergence file header
#            if save_path!=None
#                f = open(fname, "w")
#                print(f, "age (deg),T,DT")
#                for (i, rotor) in enumerate(rotors)
#                    print(f, ",RPM_$i,CT_$i,CQ_$i,eta_$i")
#                end
#                print(f, "\n")
#                close(f)
#            end
#        end
#
#        # Write rotor position and time on convergence file
#        if save_path!=None
#            f = open(fname, "a")
#            print(f, angle, ",", T, ",", DT)
#        end
#
#
#        # Plot circulation and loads distributions
#        if disp_conv
#
#            cratio = PFIELD.nt/nsteps
#            cratio = cratio > 1 ? 1 : cratio
#            clr = fcalls==0 && False ? (0,0,0) : (1-cratio, 0, cratio)
#            stl = fcalls==0 && False ? "o" : "-"
#            alpha = fcalls==0 && False ? 1 : 0.5
#
#            # Circulation distribution
#            subplot(231)
#            this_sol = []
#            for rotor in rotors
#                this_sol = vcat(this_sol, [FLOWVLM.get_blade(rotor, j).sol["Gamma"] for j in 1:rotor.B]...)
#            end
#            plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)
#
#            # Np distribution
#            subplot(232)
#            this_sol = []
#            for rotor in rotors
#                this_sol = vcat(this_sol, rotor.sol["Np"]["field_data"]...)
#            end
#            plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)
#
#            # Tp distribution
#            subplot(233)
#            this_sol = []
#            for rotor in rotors
#                this_sol = vcat(this_sol, rotor.sol["Tp"]["field_data"]...)
#            end
#            plot(1:size(this_sol,1), this_sol, stl, alpha=alpha, color=clr)
#        end
#
#        # Plot performance parameters
#        for (i,rotor) in enumerate(rotors)
#            CT, CQ = FLOWVLM.calc_thrust_torque_coeffs(rotor, rho)
#            eta = J*CT/(2*math.pi*CQ)
#
#            if disp_conv
#                subplot(234)
#                plot([angle], [CT], "$(stls[i])", alpha=alpha, color=clr)
#                subplot(235)
#                plot([angle], [CQ], "$(stls[i])", alpha=alpha, color=clr)
#                subplot(236)
#                plot([angle], [eta], "$(stls[i])", alpha=alpha, color=clr)
#            end
#
#            if save_path!=None
#                print(f, ",", rotor.RPM, ",", CT, ",", CQ, ",", eta)
#            end
#        end
#
#        if disp_conv
#            # Save figure
#            if fcalls%nsteps_savefig==0 && fcalls!=0 && save_path!=None
#                savefig(joinpath(save_path, run_name*"_convergence.png"),
#                                                            transparent=True)
#            end
#        end
#
#        # Close convergence file
#        if save_path!=None
#            print(f, "\n")
#            close(f)
#        end
#
#        fcalls += 1
#
#        return False
#return extra_runtime_function
