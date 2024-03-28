import os
import matplotlib.pyplot as plt
import argparse
import re
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Jafer\'s rudimentary plotter',
        description='...'
    )
    
    parser.add_argument('res_folder', nargs='?', help='folder containing results')
    parser.add_argument('problem', nargs='?', help='problem')
    parser.add_argument('sub_method', nargs='?', help='submethod')
    parser.add_argument('job_name', nargs='?', help='job name')
    args = parser.parse_args()
    
    orig_dir = os.getcwd()
    
    alphapat = 'alpha*'
    epspat = 'eps*'
    ref_df = pd.read_csv('parameter_reference.csv', sep=',')
    if re.match(alphapat, args.job_name):
        alpha_val = ref_df.loc[ref_df['param'] == args.job_name].iloc[0]['value']
        eps_val = 0.01
    elif re.match(epspat, args.job_name):
        alpha_val = 0.01
        eps_val = ref_df.loc[ref_df['param'] == args.job_name].iloc[0]['value']

    os.chdir(args.res_folder)
    
    fpat = '.*' + args.problem + '_.*' + args.job_name + '_.*' + args.sub_method
    files = [f for f in os.listdir('.') if re.match(fpat, f)]
    
    res_df = pd.read_csv(files[0], sep=',')
    
    plot_file_name = f'{args.problem}_{args.sub_method}_{args.job_name}_alpha{alpha_val}_eps{eps_val}.png'
    
    plt.figure()
    plt.plot(res_df['episode'], res_df['|w - w_old|'])
    plt.xlabel('episode')
    plt.ylabel('|W_new - W_old|')
    plt.title(f'{args.problem}, {args.sub_method}, alpha: {alpha_val}, eps: {eps_val}', fontsize=10)
    
    os.chdir(orig_dir)
    plt.savefig(plot_file_name)
    # plt.show()
    