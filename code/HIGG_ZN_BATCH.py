# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 13:22:49 2023

@author: User
"""
import os
from pathlib import Path
import sys

root_dir = Path('../../').resolve()
sys.path.append(root_dir.as_posix())

import numpy as np
from create_Batch_model import *
import matplotlib.pyplot as plt

from create_Batch_model import create_model, update_nex
from CADETProcess.comparison import calculate_sse
from CADETProcess.simulator import Cadet
from CADETProcess import settings


from CADETProcess.optimization import OptimizationProblem
from CADETProcess.optimization import U_NSGA3

def setup_optimization_problem():  
    # crystal phase discretization
    n_x = 100 + 2 # total number of components c_feed being the first, c_eq being the last
    x_c = 1e-6 # m
    x_max = 900e-6  # m

    # Spacing
    x_grid_n = np.logspace(np.log10(x_c), np.log10(x_max), n_x-1)  

    x_ct_n = []
    for p in range(1, n_x-1):
        x_ct_n.append((x_grid_n[p] + x_grid_n[p-1]) / 2)
    x_ct_n = np.asarray(x_ct_n)

    t_int = 2  

    # optimization problem
    simulator = Cadet(r"C:\Users\User\cadet\cadet3\bin\cadet-cli.exe") # intrinsic distribution implementation, HR in z

    optimization_problem = OptimizationProblem('HIGG_ZN_BATCH')

    # there is an ub limit, possibly in pymoo, approximately 1e16
    optimization_problem.add_variable('growth rate constant', lb=1e-9, ub=1e-5)
    optimization_problem.add_variable('growth rate exponent', lb=0.1, ub=2.0)
    optimization_problem.add_variable('nucleation rate', lb=1e12, ub=5e15)
    optimization_problem.add_variable('nucleation exponent', lb=0.5, ub=4)
    optimization_problem.add_variable('fitted ex particle count', lb=1, ub=1000)  

    
    def objective(x):
        model = create_model()
        # growth
        model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = x[0]
        model.root.input.model.unit_001.reaction_bulk.cry_g = x[1]
        
        # nucleation
        model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = x[2]
        model.root.input.model.unit_001.reaction_bulk.cry_u = x[3]
        
        n=update_nex(x[4])
    
        filename = simulator.get_tempfile_name()
    
        model.filename = filename
        model.save()
        model = simulator.run_h5(filename)
        
        # calculate residual
        residual = 0.0
        try:
            for i in range (t_int, t_int+11):
                n_x_t = model.root.output.solution.unit_001.solution_outlet[i,1:-1]
                residual += calculate_sse(n_x_t/1e14, n[i]/1e14)
        except IndexError:
            return 1e10
            
        # plotting
        fig=plt.figure(figsize = (9,4.5))
        plt.suptitle(f"kg:{np.format_float_scientific(x[0],precision=2)}, g:{np.format_float_scientific(x[1],precision=2)}, kb:{np.format_float_scientific(x[2],precision=2)}, b:{np.format_float_scientific(x[3],precision=2)}, par:{np.format_float_scientific(x[4],precision=2)}", size="xx-small")
        
        ax = fig.add_subplot(1,2,1)
        for i in range (t_int, t_int+11):
            n_x_t = model.root.output.solution.unit_001.solution_outlet[i,1:-1]
            ax.plot(x_ct_n*1e6,n_x_t)
        ax.set_xscale("log")
        ax.set_xlabel('$chord~length~/~\mu m}$')
        ax.set_ylabel('$n~/~(1/m/m^3)$')
        
        ax = fig.add_subplot(1,2,2)
        for i in range (t_int, t_int+11):
            ax.plot(x_ct_n*1e6,n[i])
        ax.set_xscale("log")
        ax.set_xlabel('$chord~length~/~\mu m}$')
        
        plt.savefig(f'{settings.working_directory}/{residual}.png', dpi=80)
        plt.close(fig)
        
        # remove .h5 file
        os.remove(model.filename)
        
        return residual
    
    optimizer = U_NSGA3()

    optimizer.n_cores = 7
    optimizer.pop_size = 21  # better: 100-200
    optimizer.n_max_gen = 5

    settings.working_directory = f'{optimization_problem.name}'

    optimization_problem.add_objective(objective)
    optimization_problem.parallelization_backend = "joblib"
    
    return optimizer, optimization_problem


print("started optimization")
if __name__ == '__main__':
    optimizer, optimization_problem = setup_optimization_problem()
    
    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=False
    )
print("ended optimization")

print(f"the best parameters are {optimization_results.x[0]}")

## plot the best results
model = create_model()
# growth
model.root.input.model.unit_001.reaction_bulk.cry_growth_rate_constant = optimization_results.x[0, 0]
model.root.input.model.unit_001.reaction_bulk.cry_g = optimization_results.x[0, 1]

# nucleation
model.root.input.model.unit_001.reaction_bulk.cry_primary_nucleation_rate = optimization_results.x[0, 2]
model.root.input.model.unit_001.reaction_bulk.cry_u = optimization_results.x[0, 3]

n=update_nex(optimization_results.x[0, 4])

filename = simulator.get_tempfile_name()
model.filename = filename
model.save()
model = simulator.run_h5(filename)
os.remove(model.filename)

# plot
fig=plt.figure(figsize = (9,4.5))
ax = fig.add_subplot(1,2,1)
for i in range (t_int, t_int+11):
    n_x_t = model.root.output.solution.unit_001.solution_outlet[i,1:-1]
    ax.plot(x_ct_n*1e6,n_x_t)
ax.set_xscale("log")
ax.set_xlabel('$chord~length~/~\mu m}$')
ax.set_ylabel('$n~/~(1/m/m^3)$')
ax.set_title("Simulated")

ax = fig.add_subplot(1,2,2)
for i in range (t_int, t_int+11):
    ax.plot(x_ct_n*1e6,n[i])
ax.set_xscale("log")
ax.set_xlabel('$chord~length~/~\mu m}$')
ax.set_title("Experimental")

plt.show()