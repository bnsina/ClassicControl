from . import linear_ortho_sgsarsa as los
from contextlib import redirect_stdout
from datetime import datetime
import torch
from gym.envs.classic_control import MountainCarEnv
from gym.envs.classic_control import CartPoleEnv
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
        
        filename = self.now_str + '_' + self.args.problem + '_' + self.args.job_name + '_' + self.args.algorithm + '_' + self.args.submethod + '.txt'
        torch.set_default_dtype(torch.float64)
        
        if self.args.problem == 'MountainCar': 
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
                with open(filename,'w') as f:
                    with redirect_stdout(f):
                        print(self.args)
                        Q, S, _ = opt.train(mountain_car, model, verbose=True, log_interval=1)
            if self.args.verbose == True:
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
                with open(filename,'w') as f:
                    with redirect_stdout(f):
                        print(self.args)
                        Q, S, _ = opt.train(cart_pole, model, verbose=True, log_interval=1)
            if self.args.verbose == True:
                Q, S, _ = opt.train(cart_pole, model, verbose=True, log_interval=1)
            
        else:
            print('Problem ' + str(self.args.problem) + ' not detected. See: classic_control -h for implemented problems.')
        
    

