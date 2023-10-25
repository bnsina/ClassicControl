from . import control
from . import multi
import argparse 
import pandas as pd
from time import perf_counter
import multiprocessing as mp

# convert param df into namespace list for pool.map(), also append verbose flag
def param_lister(input_df, retxt, num_jobs):
    ns_list = []
    for i in range(num_jobs):
        flags = pd.Series([False, retxt], index=['verbose', 'retxt'])
        pre_ns = pd.concat([input_df.iloc[i,:], flags]).to_dict()
        ns_list.append(argparse.Namespace(**pre_ns))
    return ns_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Classic Control Experiment Scheduler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=
        'Performs experiments specified by an input CSV file with columns: \n\n' +
        '{job_name, problem, algorithm, featurizer, submethod, alpha, constant_step,\n' + 
        'gamma, num_episodes, max_steps, max_eps, min_eps, order, seed, cvlen}\n\n'+
        'Jobs are performed sequentially by default, with an option for \n' + 
        'multiprocessing for long or numerous experiments.\n\n' + 
        'If -v (verbose) is not set, output of each job is stored in a file named\n' + 
        '[DATE]_[TIME]_[problem]_[job_name]_[algorithm]_[submethod].[ext]\n' +
        'where [ext] is CSV (default) of TXT (enabled by -t flag)' +
        'If -v is set, output is printed in the console (not compatible with multiprocessing).\n\n' +
        '-------Implemented-------\n' +
        'Problems:\n' + 
        '   MountainCar\n' + 
        '   CartPole\n' +
        '   Trajectory\n' +
        'Algorithms->Submethods:\n' + 
        '   SemiGradientSARSAOrtho\n' + 
        '       -> Original, Procrustes, CurvilinearInverse, CurvilinearNoInverse, QR\n' +
        'Step Control\n' + 
        '    Armijo\n' + 
        'Featurizers:\n' + 
        '   Linear\n\n' +
        '--To be implemented:--\n'          
    )
    parser.add_argument('filename', nargs='?', help='filename/path of parameter CSV')
    parser.add_argument('-m', '--multi', dest='multi', action='store_true', help='enable multiprocessing')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='print output to console, forces single-processing!')
    parser.add_argument('-t', '--txt', dest='retxt', action='store_true', help='send output to TXT instead of CSV')
    parser.add_argument('-z', '--visual', dest='visual', action='store_true', help='visualize policy (only for Trajectory CSV mode)')
    args = parser.parse_args()
    
    input_df = pd.read_csv(args.filename, sep=',')
    num_jobs, _ = input_df.shape
    
    vflag = pd.Series([args.verbose, args.retxt, args.visual], index=['verbose', 'retxt', 'visual'])
    # -v overrides -m and forces single-processing
    if args.verbose == True:
        args.multi = False
    
    # no multiprocessing: do jobs sequentially
    if args.multi == False:
        t_initial = perf_counter()
        
        # convert param df into namespace list, also append verbose flag
        for i in range(num_jobs):
            pre_ns = pd.concat([input_df.iloc[i,:], vflag]).to_dict()
            job = control.controller(argparse.Namespace(**pre_ns))
            job.run()
            
        t_final = perf_counter()
        print("Time elapsed: " + str(t_final - t_initial) + ' secs')
    
    # multiprocessing
    if args.multi == True:
        if mp.cpu_count() > 1:
            proc_size = mp.cpu_count() - 1 # max recruited processes
        else:
            proc_size = 1
        
        t_initial = perf_counter()
        
        param_list = param_lister(input_df, args.retxt, num_jobs)
        
        with mp.Pool(processes=proc_size) as pool:
            pool.map(multi.runner, param_list)
        
        t_final = perf_counter()
        print("Time elapsed: " + str(t_final - t_initial) + ' secs')