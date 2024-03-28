from . import linear_ortho_sgsarsa as los
from contextlib import redirect_stdout
from datetime import datetime
from os import path, getcwd
import torch
import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
from gym.envs.classic_control import MountainCarEnv
from gym.envs.classic_control import CartPoleEnv
from gnflow3.envs import TrajectoryEnv, TrajectoryEnvVisualizersFunctionApproximation
import gnflow3.featurizers
import gnflow3.models as models
from gnflow3.utils import set_seed
from gnflow3.utils import discretize_box_state, get_argument_parser
from gnflow3.utils import extract_data, insert_data


class controller:
    
    def __init__(self, namespace):
        self.args = namespace
        self.now = datetime.now()
        self.now_str = self.now.strftime('%m-%d-%Y_%H-%M')

    # initial weights
    def initialize_weights(self, model):
        ns, na = torch.Tensor.size(model.w)
        q, r = torch.linalg.qr(torch.rand(ns, na), mode='reduced')
        q = torch.Tensor(q)
        insert_data(model, torch.reshape(q, (-1,)))  
    
    # algorithm switch
    def choose_algo(self, **kwargs):
        if self.args.algorithm == 'SemiGradientSARSAOrtho':
            return los.LinearSemiGradientSARSAOrtho(**kwargs)
        else:
            print('Algorithm ' + str(self.args.algorithm) + ' not found. See: classic_control -h for implemented algorithms')
    
    # featurizer switch     
    def choose_featurizer(self):
        if self.args.featurizer == 'Linear':
            return gnflow3.featurizers.PolynomialFeaturizer(order=self.args.order)

    def run(self):
        
        print(self.args)
        print('PyTorch version: ' + torch.__version__)
        
        filename = self.now_str + '_' + self.args.problem + '_' + self.args.job_name + '_' + self.args.algorithm + '_' + self.args.submethod 
        
        # TEMP: set 64bit floating
        torch.set_default_dtype(torch.float64)
        
        if self.args.problem == 'temp-PG':
            mountain_car = MountainCarEnv()
            # featurizer
            
            num_actions = mountain_car.action_space.n
            num_state = mountain_car.observation_space.shape[0]
            # model needs to return a value and a gradient
            
            set_seed(self.args.seed, mountain_car)
            
            # self.initialize_weights(model)
            
            # opt = self.choose_algo # policy gradient
            # have main loop and run episode in the same thing,
            # pass args to this.
            
            
            pass    
        
        elif self.args.problem == 'MountainCar': 
            mountain_car = MountainCarEnv()
            featurizer = self.choose_featurizer()

            num_actions = mountain_car.action_space.n
            num_state = mountain_car.observation_space.shape[0]
            model = models.LinearModel(featurizer, num_state, num_actions)

            set_seed(self.args.seed, mountain_car) 
            
            self.initialize_weights(model)
            
            opt = self.choose_algo(**vars(self.args))
            
            # write to file if not verbose
            if self.args.verbose == False:
                
                if self.args.retxt == True:
                    with open(filename + '.txt','w') as f:
                        with redirect_stdout(f):
                            print(self.args)
                            Q, S, _ = opt.train(mountain_car, model, verbose=True, log_interval=1)            
                else:
                    Q, S, _ = opt.train(mountain_car, model, verbose=False, log_interval=1)
                    
                    mc_res = pd.DataFrame(opt.history['value'], columns=opt.history['header'])
                    pd.DataFrame.to_csv(mc_res, path.join(getcwd(), filename + '.csv'))
            
            if self.args.verbose == True:
                print(self.args.job_name)
                Q, S, _ = opt.train(mountain_car, model, verbose=True, log_interval=1)
            
        elif self.args.problem == 'CartPole':  
            cart_pole = CartPoleEnv()
            featurizer = self.choose_featurizer()

            num_actions = cart_pole.action_space.n
            num_state = cart_pole.observation_space.shape[0]
            model = models.LinearModel(featurizer, num_state, num_actions)

            set_seed(self.args.seed, cart_pole)
            
            self.initialize_weights(model)
            
            opt = self.choose_algo(**vars(self.args))
            
            # write to file if not verbose
            if self.args.verbose == False:
                
                if self.args.retxt == True:
                    with open(filename + '.txt','w') as f:
                        with redirect_stdout(f):
                            print(self.args)
                            Q, S, _ = opt.train(cart_pole, model, verbose=True, log_interval=1)            
                else:
                    Q, S, _ = opt.train(cart_pole, model, verbose=False, log_interval=1)
                    
                    mc_res = pd.DataFrame(opt.history['value'], columns=opt.history['header'])
                    pd.DataFrame.to_csv(mc_res, path.join(getcwd(), filename + '.csv'))
                            
            if self.args.verbose == True:
                print(self.args.job_name)
                Q, S, _ = opt.train(cart_pole, model, verbose=True, log_interval=1)
        
        elif self.args.problem == 'Trajectory':  
            traj = TrajectoryEnv(grid_size=array([16, 16]))
            featurizer = self.choose_featurizer()

            num_actions = traj.action_space.n
            num_state = traj.observation_space.shape[0]
            model = models.LinearModel(featurizer, num_state, num_actions)

            set_seed(self.args.seed, traj)
           
            self.initialize_weights(model)
           
            opt = self.choose_algo(**vars(self.args))
           
            # write to file if not verbose
            if self.args.verbose == False:
               
                if self.args.retxt == True:
                    with open(filename + '.txt','w') as f:
                        with redirect_stdout(f):
                            print(self.args)
                            Q, S, _ = opt.train(traj, model, verbose=True, log_interval=1)            
                else:
                    Q, S, _ = opt.train(traj, model, verbose=False, log_interval=1)
                   
                    mc_res = pd.DataFrame(opt.history['value'], columns=opt.history['header'])
                    pd.DataFrame.to_csv(mc_res, path.join(getcwd(), filename + '.csv'))
            
            if self.args.verbose == True:
                print(self.args.job_name)
                Q, S, _ = opt.train(traj, model, verbose=True, log_interval=1)
                
                # print(opt.best)
                # print(model.w)
                # print(f'Policy tester: {opt.policy(model, array([-0.8125 -0.9375]), 0)}')
                opt.model = model
                print(f'{opt.model}')         
                # viz test
                visualizer = TrajectoryEnvVisualizersFunctionApproximation(traj, opt)

                # visualizer.plot_world(show_colorbar=True, show_start=True, show_end=True, show_legend=True, show_axis=True)
                # plt.show()

                visualizer.plot_greedy_path(show_colorbar=True, show_start=True, show_end=True)
                plt.show()

                # visualizer.plot_greedy_policy(show_colorbar=True, show_start=True, show_end=True)
                # plt.show()
            
        else:
            print('Problem ' + str(self.args.problem) + ' not detected. See: classic_control -h for implemented problems.')
        
    





# #def choose_exploration_parameters(range_eps: tuple = (1.0, 1e-4), num_episodes: tuple = (100,)):

#     assert len(range_eps) == len(num_episodes) + 1

#     # sort range
#     range_eps = np.sort(range_eps)
#     range_eps = range_eps[::-1]

#     eps = np.empty(0)
#     for i, n in enumerate(num_episodes):
#         tmp_eps = np.logspace(np.log10(range_eps[i]), np.log10(range_eps[i + 1]), num=n)

#         if i > 0:
#             eps = np.concatenate((eps, tmp_eps[1:]))
#         else:
#             eps = np.concatenate((eps, tmp_eps))

#     return eps

# #if __name__ == "__main__":
#     eps = choose_exploration_parameters((1, 1e-4), (5,))
#     print(eps)

#     eps = choose_exploration_parameters((1, 1e-2, 1e-4), (2, 2))
#     print(eps)