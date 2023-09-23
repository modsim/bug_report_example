# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:43:34 2023

@author: User
"""
import pandas as pd
import numpy as np

## Blaze grid
x_min = 1e-6 # m
x_max = 900e-6 # m
n_bins = 100

x_grid = np.logspace(np.log10(x_min),np.log10(x_max),n_bins+1)

# midpoint, arithmetic mean
x_ct = []
for i in range (n_bins):
    x_ct.append(0.5*x_grid[i]+0.5*x_grid[i+1])
x_ct = np.asarray(x_ct)


data_dir = r"C:\Users\User\blaze\6.6.23\230606 higg_zn_batchExportOriginal.xlsx"

sheet_LW = "E2E-ChannelData-LW"

df_LW = pd.read_excel(data_dir, sheet_name=sheet_LW, header=0, skiprows=4)
df_LW.drop(columns=df_LW.columns[-1],axis=1,inplace=True) # drop the last column, which is empty

# slice data [row,col]
time = df_LW.iloc[:,0:1].to_numpy() # s
mean_size = df_LW.iloc[:,2:3].to_numpy() # um
particle_count = df_LW.iloc[:,3:4].to_numpy() # 1
number_distri = df_LW.iloc[:,4:].to_numpy() # 1

# to find the starting point, set a value so that the increase is bigger than the set value, it will be treated as the addition point
threshold = 10000
for i in range (0,len(time)):
    if (particle_count[i+1] - particle_count[i]) > threshold:
        addition_point_row = i+1
        break

print("successfully get experimental results")
# remember to delete the background
time_frame = 16  ## plot how many time points after addition

def normalize_cld(cld):
    total_cld = np.sum(cld)
    normalized_cld = []
    for i in range (0, len(cld)):
        normalized_cld.append(cld[i] / total_cld)
    return np.asarray(normalized_cld)

def calculate_n(psd, x_ct, reactor_dimension):
    # calculate the number density 1/m cry/m rea
    n = []
    for i in range (0, len(psd)):
        n.append(psd[i]/x_ct[i]/reactor_dimension)
    n = np.asarray(n)   # the first m is the particle size, the second m is the reactor length
    return n

from scipy.interpolate import interp1d
def normalize_extrap_n(n, x_ct_old, x_ct_new):
    area = np.trapz(n, x_ct_old)
    n = n/area
    spl = interp1d(x_ct_old, n, kind="quadratic", fill_value="extrapolate")
    normalized_n = spl(x_ct_new)
    return normalized_n

## convert the cld to psd
from cld_psd_utils import get_A, cld_to_psd

# imageJ input
cir = 0.84

# reactor input
reactor_volume = 0.25*900**2*np.pi*330 /1e18 # m^3, this might also be inaccurate

# get A
A = get_A(x_grid, cir)
print("successfully get A")

def update_nex(count):
    total_particle_count = count # 26666, this is leveraged
    psd = []
    n = []
    for i in range (0, time_frame):
        r_psd, res = cld_to_psd(A, normalize_cld(number_distri[addition_point_row+i]), fre=0.15, order=1)
        psd.append(r_psd * total_particle_count)
        n.append(calculate_n(r_psd * total_particle_count, x_ct, reactor_volume))
    return n

n = update_nex(3.8)

# batch experiments
from cadet import Cadet as CADETPython

# enable Jacobian
jacobian = 1

# time resolution
time_resolution = 60 +1

# feed
c_feed = 9.9*0.8
c_eq = 0.707

# crystal phase discretization
n_x = 100 + 2 # total number of components c_feed being the first, c_eq being the last
x_c = 1e-6 # m
x_max = 900e-6  # m

# simulation time
cycle_time = 60 # s

# volume
v_reactor = 35e-6 # m^3

# Spacing
x_grid_n = np.logspace(np.log10(x_c), np.log10(x_max), n_x-1)  

x_ct_n = []
for p in range(1, n_x-1):
    x_ct_n.append((x_grid_n[p] + x_grid_n[p-1]) / 2)
x_ct_n = np.asarray(x_ct_n)

# PDF
t_int = 2                                                  # discard the first 2 seconds
#nPDF = normalize_extrap_n(n[t_int], x_ct*1e6, x_ct_n*1e6)  # the change in particle count does not change the nPDF, no need to update

# Boundary conditions
boundary_c = []
for p in range(n_x):
    if p == 0:
        boundary_c.append(c_feed)
    elif p == n_x-1:
        boundary_c.append(c_eq)
    else:
        boundary_c.append(0)
boundary_c = np.asarray(boundary_c)

# Initial conditions
initial_c = []
for k in range(n_x):
    if k == n_x-1:
        initial_c.append(c_eq)
    elif k==0:
        initial_c.append(c_feed)
    else:
        initial_c.append(0)
initial_c = np.asarray(initial_c)

def create_model():
    model = CADETPython()

    # number of unit operations
    model.root.input.model.nunits = 2

    #inlet model
    model.root.input.model.unit_000.unit_type = 'INLET'
    model.root.input.model.unit_000.ncomp = n_x
    model.root.input.model.unit_000.inlet_type = 'PIECEWISE_CUBIC_POLY'

    #time sections
    model.root.input.solver.sections.nsec = 1
    model.root.input.solver.sections.section_times = [0.0, 8000,]   # s
    model.root.input.solver.sections.section_continuity = []

    model.root.input.model.unit_000.sec_000.const_coeff = n_x*[0.0,] #boundary_c 
    model.root.input.model.unit_000.sec_000.lin_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.quad_coeff = n_x*[0.0,]
    model.root.input.model.unit_000.sec_000.cube_coeff = n_x*[0.0,]

    # CSTR
    model.root.input.model.unit_001.unit_type = 'CSTR'
    model.root.input.model.unit_001.ncomp = n_x
    model.root.input.model.unit_001.use_analytic_jacobian = jacobian # don't change now
    model.root.input.model.unit_001.init_c = initial_c
    model.root.input.model.unit_001.init_volume = v_reactor
    model.root.input.model.unit_001.porosity = 1
    model.root.input.model.unit_001.adsorption_model = 'NONE'

    # crystallization reaction
    model.root.input.model.unit_001.reaction_model = 'CRYSTALLIZATION'
    model.root.input.model.unit_001.reaction_bulk.cry_bins = x_grid
    model.root.input.model.unit_001.reaction_bulk.cry_nuclei_mass_density = 1.2e3
    model.root.input.model.unit_001.reaction_bulk.cry_vol_shape_factor = 0.524

    ## nucleation
    model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = 5
    model.root.input.model.unit_001.reaction_bulk.cry_u = 10.0

    model.root.input.model.unit_001.reaction_bulk.cry_secondary_nucleation_rate = 4e8
    model.root.input.model.unit_001.reaction_bulk.cry_b = 2.0
    model.root.input.model.unit_001.reaction_bulk.cry_k = 1.0

    ## growth
    model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = 0.0e-6
    model.root.input.model.unit_001.reaction_bulk.cry_growth_constant = 0
    model.root.input.model.unit_001.reaction_bulk.cry_a = 1.0
    model.root.input.model.unit_001.reaction_bulk.cry_g = 1.0
    model.root.input.model.unit_001.reaction_bulk.cry_p = 0.0
    model.root.input.model.unit_001.reaction_bulk.cry_growth_dispersion_rate = 0 # 1.5e-11
    model.root.input.model.unit_001.reaction_bulk.cry_growth_scheme_order = 2 # can only be 1, 2, 3, 4

    ## Outlet
    model.root.input.model.unit_002.unit_type = 'OUTLET'
    model.root.input.model.unit_002.ncomp = n_x

    # Connections
    Q = 0 # volumetric flow rate

    model.root.input.model.connections.nswitches = 1
    model.root.input.model.connections.switch_000.section = 0
    model.root.input.model.connections.switch_000.connections = [
        0, 1, -1, -1, Q,
        1, 2, -1, -1, Q,
    ]  # Q, volumetric flow rate 

    # numerical solver configuration
    model.root.input.model.solver.gs_type = 1
    model.root.input.model.solver.max_krylov = 0
    model.root.input.model.solver.max_restarts = 10
    model.root.input.model.solver.schur_safety = 1e-8

    # Number of cores for parallel simulation
    model.root.input.solver.nthreads = 1

    # Tolerances for the time integrator
    model.root.input.solver.time_integrator.abstol = 1e-6
    model.root.input.solver.time_integrator.algtol = 1e-10
    model.root.input.solver.time_integrator.reltol = 1e-6
    model.root.input.solver.time_integrator.init_step_size = 1e-10
    model.root.input.solver.time_integrator.max_steps = 1000000

    # Return data
    model.root.input['return'].split_components_data = 0
    model.root.input['return'].split_ports_data = 0
    model.root.input['return'].unit_000.write_solution_bulk = 1
    model.root.input['return'].unit_000.write_solution_inlet = 1
    model.root.input['return'].unit_000.write_solution_outlet = 1
    model.root.input['return'].unit_000.write_sens_outlet=1
    model.root.input['return'].unit_000.write_sens_bulk=1

    # Copy settings to the other unit operations
    model.root.input['return'].unit_001 = model.root.input['return'].unit_000
    model.root.input['return'].unit_002 = model.root.input['return'].unit_000

    # Solution times
    model.root.input.solver.user_solution_times = np.linspace(0, cycle_time, time_resolution)
    
    return model