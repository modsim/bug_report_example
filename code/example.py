import numpy as np

from CADETProcess.comparison import Comparator
from CADETProcess.optimization import OptimizationProblem, U_NSGA3
from CADETProcess.processModel import ComponentSystem, FlowSheet, Inlet, Outlet, LumpedRateModelWithPores, Process
from CADETProcess.reference import ReferenceIO
from CADETProcess.simulator import Cadet

if __name__ == '__main__':
    data = np.loadtxt('../data/non_pore_penetrating_tracer.csv', delimiter=',')

    time_experiment = data[:, 0]
    c_experiment = data[:, 1]

    tracer_peak = ReferenceIO(
        'Tracer Peak', time_experiment, c_experiment
    )

    component_system = ComponentSystem(['Non-penetrating Tracer'])

    feed = Inlet(component_system, name='feed')
    feed.c = [0.0005]

    eluent = Inlet(component_system, name='eluent')
    eluent.c = [0]

    column = LumpedRateModelWithPores(component_system, name='column')

    column.length = 0.1
    column.diameter = 0.0077
    column.particle_radius = 34e-6

    column.axial_dispersion = 1e-8
    column.bed_porosity = 0.3
    column.particle_porosity = 0.8
    column.film_diffusion = [0]

    outlet = Outlet(component_system, name='outlet')

    flow_sheet = FlowSheet(component_system)

    flow_sheet.add_unit(feed)
    flow_sheet.add_unit(eluent)
    flow_sheet.add_unit(column)
    flow_sheet.add_unit(outlet)

    flow_sheet.add_connection(feed, column)
    flow_sheet.add_connection(eluent, column)
    flow_sheet.add_connection(column, outlet)

    Q_ml_min = 0.5  # ml/min
    Q_m3_s = Q_ml_min / (60 * 1e6)
    V_tracer = 50e-9  # m3

    process = Process(flow_sheet, 'Tracer')
    process.cycle_time = 15 * 60

    process.add_event(
        'feed_on',
        'flow_sheet.feed.flow_rate',
        Q_m3_s, 0
    )
    process.add_event(
        'feed_off',
        'flow_sheet.feed.flow_rate',
        0,
        V_tracer / Q_m3_s
    )

    process.add_event(
        'feed_water_on',
        'flow_sheet.eluent.flow_rate',
        Q_m3_s,
        V_tracer / Q_m3_s
    )

    process.add_event(
        'eluent_off',
        'flow_sheet.eluent.flow_rate',
        0,
        process.cycle_time
    )

    simulator = Cadet()

    comparator = Comparator()
    comparator.add_reference(tracer_peak)
    comparator.add_difference_metric(
        'NRMSE', tracer_peak, 'outlet.outlet',
    )

    optimization_problem = OptimizationProblem('bed_porosity_axial_dispersion')

    optimization_problem.add_evaluation_object(process)

    optimization_problem.add_variable(
        name='bed_porosity', parameter_path='flow_sheet.column.bed_porosity',
        lb=0.1, ub=0.6,
        transform='auto'
    )

    optimization_problem.add_variable(
        name='axial_dispersion', parameter_path='flow_sheet.column.axial_dispersion',
        lb=1e-10, ub=0.1,
        transform='auto'
    )

    optimization_problem.add_evaluator(simulator)

    optimization_problem.add_objective(
        comparator,
        n_objectives=comparator.n_metrics,
        requires=[simulator]
    )


    def callback(simulation_results, individual, evaluation_object, callbacks_dir='./'):
        comparator.plot_comparison(
            simulation_results,
            file_name=f'{callbacks_dir}/{individual.id}_{evaluation_object}_comparison.png',
            show=False
        )


    optimization_problem.add_callback(callback, requires=[simulator])

    optimizer = U_NSGA3()
    optimizer.n_cores = 10

    optimization_results = optimizer.optimize(
        optimization_problem,
        use_checkpoint=False
    )
