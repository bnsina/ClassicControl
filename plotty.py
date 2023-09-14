import os
import matplotlib.pyplot as plt
import argparse
import re
import pandas as pd

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        prog='Rudimentary plotter',
        description='...'
    )
    
    parser.add_argument('res_folder', nargs='?', help='folder containing results')
    parser.add_argument('job_name', nargs='?', help='job name')
    parser.add_argument('sub_method', nargs='?', help='submethod')
    args = parser.parse_args()
    
    os.chdir(args.res_folder)
    
    fpat = '.*' + args.job_name + '_.*' + args.sub_method
    files = [f for f in os.listdir('.') if re.match(fpat, f)]
    
    res_df = pd.read_csv(files[0], sep=',')
    
    plt.figure()
    plt.plot(res_df['episode'], res_df['total_reward'])
    plt.xlabel('episode')
    plt.ylabel('return')
    plt.ylim(-9001, 0)
    plt.title(f'Submethod: {args.sub_method}, Job: {args.job_name}')
    plt.show()
    