import numpy as np
import torch
import math
from copy import deepcopy
from gnflow3.algorithms import LinearFunctionApproximationAlgorithm
from gnflow3.utils import extract_data, insert_data

class LinearSemiGradientSARSAOrtho(LinearFunctionApproximationAlgorithm):

    def __init__(self, *args, **kwargs):
        super(LinearSemiGradientSARSAOrtho, self).__init__(*args, **kwargs)
        self.history['header'] += ('iter', 'alpha', '|w - w_old|', '|w^T w - I|')
        self.history['format'] += '{:<15d}{:<15.2e}{:<15.2e}{:<15.2e}'
        self.submethod = kwargs['submethod']
    
    # submethod switch
    def new_weights(self, model, theta, alpha, D, D_grad):
        ns, na = torch.Tensor.size(model.w)
        
        if self.submethod == 'Original':
            dtheta = -alpha * D * D_grad
            insert_data(model, theta + dtheta)
            
        elif self.submethod == 'Procrustes':
            dtheta = -alpha * D * D_grad
            w_rs = torch.reshape(theta + dtheta, (ns, na))
            U, S, V = torch.svd(w_rs)
            w_ortho = torch.matmul(U, V.T)
            insert_data(model, torch.reshape(w_ortho, (-1,)))
            
        elif self.submethod == 'CurvilinearInverse':
            w_rs = torch.reshape(theta, (ns, na))
            dq_rs = torch.reshape(D_grad, (ns, na))
            U = torch.cat((dq_rs, w_rs), dim=1)
            Vt = torch.cat((w_rs, -dq_rs), dim=1).t()
            t = 1 
            inv = torch.inverse(torch.eye(2*na, 2*na) + (t/2) * torch.matmul(Vt, U))
            w_ortho = w_rs - t * torch.linalg.multi_dot((U, inv, Vt, w_rs))
            insert_data(model, torch.reshape(w_ortho, (-1,)))
        
        elif self.submethod == 'CurvilinearNoInverse':
            w_rs = torch.reshape(theta, (ns, na))
            dq_rs = torch.reshape(D_grad, (ns, na))
            Afrag =  torch.matmul(dq_rs, w_rs.t())
            A = Afrag - Afrag.t()
            t = alpha
            LHS = torch.eye(ns) + (t/2) * A
            RHS = torch.matmul(torch.eye(ns) - (t/2) * A, w_rs)
            w_ortho = torch.linalg.solve(LHS, RHS)
            insert_data(model, torch.reshape(w_ortho, (-1,)))
        
        elif self.submethod == 'QR':
            dtheta = -alpha * D * D_grad
            w_rs = torch.reshape(theta + dtheta, (ns, na))
            q, _ = torch.linalg.qr(w_rs, mode='reduced')
            w_ortho = torch.Tensor(q)
            insert_data(model, torch.reshape(w_ortho, (-1,)))
        
        else:
            print('Submethod ' + str(self.submethod) + ' not found. See: classic_control -h for implemented submethods.')

    def run_episode(self, env, model, ieps):

        # get featurizer scheme
        ns, na = torch.Tensor.size(model.w)

        eps = self.eps[ieps]

        # store weights before running episode
        # be sure model.w starts with orthonormal weights!
        w_old = torch.clone(model.w.data)

        # initialize state and action
        state = np.copy(env.reset()[0])

        # choose action
        action = self.policy(model, state, eps)

        if self.store_steps and ((ieps + 1) % self.store_every == 0 or ieps == self.num_episodes - 1 or ieps == 0):
            self.step_info['w_list'][ieps] = torch.clone(model.w.data)

        reward, done, total_reward, termination_flag, _ = self.initialize_episode()

        while not done:

            # step
            next_state, reward, done, _, _ = env.step(action)

            # algorithm termination - too many steps or outside observation space
            reward, done, termination_flag = self.episode_termination_conditions(env, next_state, reward, done)

            # update weights
            if not self.constant_step:
                alpha = self.alpha / (1.0 + math.sqrt(self.iter_count + 1))
            else:
                alpha = self.alpha

            # part of Bellman error
            D = reward - model(state, action)

            # form gradient (does not depend on next state)
            model.zero_grad()
            D.backward()
            theta = extract_data(model, 'data')
            D_grad = extract_data(model, 'grad')

            if not done:
                next_action = self.policy(model, next_state, eps)
                D += self.gamma * model(next_state, next_action)

                state = deepcopy(next_state)
                action = deepcopy(next_action)
                self.iter_count += 1

            # form new search direction
            self.new_weights(model, theta, alpha, D, D_grad)

            if done:
                self.extra_info = [self.iter_count, alpha, torch.norm(model.w.data - w_old).item(),
                                   torch.norm(model.w.data.T @ model.w.data - torch.eye(model.w.data.shape[1]))]

            # update steps
            total_reward += reward
            self.num_steps += 1

        # get info, stored as dict
        info = self.episode_info(total_reward, action, reward, termination_flag)
        return info