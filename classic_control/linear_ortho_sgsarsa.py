import numpy as np
import torch
import math
from copy import deepcopy
from gnflow3.algorithms import LinearFunctionApproximationAlgorithm
from gnflow3.utils import extract_data, insert_data
from time import perf_counter

class LinearSemiGradientSARSAOrtho(LinearFunctionApproximationAlgorithm):

    def __init__(self, *args, **kwargs):
        super(LinearSemiGradientSARSAOrtho, self).__init__(*args, **kwargs)
        self.history['header'] += ('iter', 'alpha', '|w - w_old|', '|w^T w - I|', 'rel_dist')
        self.history['format'] += '{:<15d}{:<15.2e}{:<15.2e}{:<15.2e}'
        self.submethod = kwargs['submethod']
        self.cvlen = kwargs['cvlen']
        self.ns = 0
        self.na = 0
    
    ## STEP CONTROL WORKFLOW
    # run_episode -> new_weights -> get_cvlen ----(yes) ---> {submethod}_traverse([params], stepsize) -> OUTPUT NEW WEIGHTS
    #                                          |
    #                                          |.constant step size?
    #                                          |
    #                                          |--(no ) ---> step_control -> OUTPUT NEW WEIGHTS
    #                                                        |         ^
    #                                                        |         | 
    #                                                        {submethod}_traverse([params], currentstepsize)
    #
    
    
## STEP CONTROL ##########
    def get_cvlen(self, alpha): 
        
        # print('here2')
        
        if self.cvlen == 'alpha':
            return alpha
        
        elif self.cvlen == 'armijo':
            return 'armijo'

        elif type(self.cvlen) is float:
            return self.cvlen
        
        else: 
            print(f'{self.cvlen} not recognized')
            return None
    
    # step size control
    def step_control(self, stepcn, func, **kwargs):
        
        # print('here3')
        
        model, D = kwargs['model_params']
        state, action, reward = kwargs['sar']
        
        if stepcn == 'armijo':
            alpha = 1.0
            rho = 0.2
            
            D_old = D.detach().clone()
            
            # unpack traverse function specific things
            if func.__name__ == 'cvinverse_traverse':
                params = kwargs['cvi_frags']
                w_rs, dq_rs = params
                Afrag =  torch.matmul(dq_rs, w_rs.t())
                A = Afrag - Afrag.t()
                part = rho * torch.inner(torch.reshape(dq_rs, (-1,)), torch.reshape(-1 * torch.matmul(A, w_rs), (-1,)))
                
            elif func.__name__ == 'cvnoinverse_traverse':
                params = kwargs['cvni_frags']
                w_rs, dq_rs = params
                Afrag =  torch.matmul(dq_rs, w_rs.t())
                A = Afrag - Afrag.t()
                part = rho * torch.inner(torch.reshape(dq_rs, (-1,)), torch.reshape(-1 * torch.matmul(A, w_rs), (-1,)))
                params.append(A)
            
            else:
                print('ERROR: invalid func.__name__')
            
            maxiter = 100
            iter = 0
            
            while True:
                # print(iter)
                
                insert_data(model, torch.reshape(func(params, alpha), (-1,)))
                D_new = reward - model(state, action)
                
                # maxiter stopgap
                iter += 1
                if iter == maxiter:
                    w_ortho = model.w.data
                    break
                
                # armijo condition
                if D_new <= D_old + alpha * part:
                    # print('yes')
                    w_ortho = model.w.data
                    break
                else:
                    # print('no')
                    alpha = alpha/2            
        else:
            pass
        
        # reset model (temporary, may not be needed)
        insert_data(model, torch.reshape(w_rs, (-1,)))
        
        return w_ortho
    
    
## CURVILINEAR TRAVERSERS ##########
    # curvilinear inverse evaluator
    def cvinverse_traverse(self, params, t):
        
        w_rs, dq_rs = params
        
        U = torch.cat((dq_rs, w_rs), dim=1)
        Vt = torch.cat((w_rs, -dq_rs), dim=1).t()
        inv = torch.inverse(torch.eye(2*self.na, 2*self.na) + (t/2) * torch.matmul(Vt, U))
        
        return (w_rs - t * torch.linalg.multi_dot((U, inv, Vt, w_rs)))
    
    # curvilinear no inverse evaluator
    def cvnoinverse_traverse(self, params, t):
        
        w_rs, dq_rs, A = params
        
        LHS = torch.eye(self.ns) + (t/2) * A
        RHS = torch.matmul(torch.eye(self.ns) - (t/2) * A, w_rs)
        
        return (torch.linalg.solve(LHS, RHS))
        
    
    
## SUBMETHOD SWTICH ##########
    def new_weights(self, model, theta, alpha, D, D_grad, reward, state, action):
        
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
            dtheta = D * D_grad
            w_rs = torch.reshape(theta, (self.ns, self.na))
            dq_rs = torch.reshape(dtheta, (self.ns, self.na))
            stepcn = self.get_cvlen(alpha)
            
            # print(stepcn)
            
            # note: step_control calling card: 'cvi_frags'
            if type(stepcn) is str:
                w_ortho = self.step_control(stepcn, self.cvinverse_traverse, 
                                            model_params = [model, D], 
                                            sar = [state, action, reward], 
                                            cvi_frags = [w_rs, dq_rs])
            else:
                w_ortho = self.cvinverse_traverse([w_rs, dq_rs], stepcn)
            
            insert_data(model, torch.reshape(w_ortho, (-1,)))
            ## old stuff
            # t = self.get_cvlen(alpha)
            # dtheta = D * D_grad
            # w_rs = torch.reshape(theta, (ns, na))
            # dq_rs = torch.reshape(dtheta, (ns, na))
            # U = torch.cat((dq_rs, w_rs), dim=1)
            # Vt = torch.cat((w_rs, -dq_rs), dim=1).t()
            # inv = torch.inverse(torch.eye(2*na, 2*na) + (t/2) * torch.matmul(Vt, U))
            # w_ortho = w_rs - t * torch.linalg.multi_dot((U, inv, Vt, w_rs))
            # insert_data(model, torch.reshape(w_ortho, (-1,)))
            
        elif self.submethod == 'CurvilinearNoInverse':
            dtheta = D * D_grad
            w_rs = torch.reshape(theta, (ns, na))
            dq_rs = torch.reshape(dtheta, (ns, na))            
            stepcn = self.get_cvlen(alpha)
            
            # note: step_control calling card: 'cvni_frags'
            if type(stepcn) is str:
                w_ortho = self.step_control(stepcn, self.cvnoinverse_traverse,
                                            model_params = [model, D],
                                            sar = [state, action, reward],
                                            cvni_frags = [w_rs, dq_rs]) # differentiated in case want to pass additional things
            
            else:
                Afrag =  torch.matmul(dq_rs, w_rs.t())
                A = Afrag - Afrag.t()
                w_ortho = self.cvnoinverse_traverse([w_rs, dq_rs, A], stepcn)
            
            insert_data(model, torch.reshape(w_ortho, (-1,)))
            ## old stuff
            # t = self.get_cvlen(alpha)
            # dtheta = D * D_grad
            # w_rs = torch.reshape(theta, (ns, na))
            # dq_rs = torch.reshape(dtheta, (ns, na))
            # Afrag =  torch.matmul(dq_rs, w_rs.t())
            # A = Afrag - Afrag.t()
            # LHS = torch.eye(ns) + (t/2) * A
            # RHS = torch.matmul(torch.eye(ns) - (t/2) * A, w_rs)
            # w_ortho = torch.linalg.solve(LHS, RHS)
            # insert_data(model, torch.reshape(w_ortho, (-1,)))
        
        elif self.submethod == 'QR':
            dtheta = -alpha * D * D_grad
            w_rs = torch.reshape(theta + dtheta, (ns, na))
            q, _ = torch.linalg.qr(w_rs, mode='reduced')
            w_ortho = torch.Tensor(q)
            insert_data(model, torch.reshape(w_ortho, (-1,)))
        
        else:
            print('Submethod ' + str(self.submethod) + ' not found. See: classic_control -h for implemented submethods.')



## RUNNER ##########
    def run_episode(self, env, model, ieps):

        # get featurizer scheme
        self.ns, self.na = torch.Tensor.size(model.w)

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
            # t_init = perf_counter()
            
            self.new_weights(model, theta, alpha, D, D_grad, reward, state, action)
            
            # t_fin = perf_counter()
            # print(f'Inner time: {t_fin - t_init}')

            # euclidean angle between stacked column standard and non-standard weight
            stan_update = (theta - alpha * D * D_grad).detach().numpy()
            curr_update = extract_data(model, 'data').detach().numpy()
            rel_dist = np.arccos(np.dot(stan_update, curr_update)/(np.linalg.norm(stan_update)*np.linalg.norm(curr_update)))
            
            wtw_minus_eye = torch.norm(model.w.data.T @ model.w.data - torch.eye(model.w.data.shape[1]))
            
            if done:
                self.extra_info = [self.iter_count, alpha, torch.norm(model.w.data - w_old).item(),
                                   wtw_minus_eye.detach().numpy(), rel_dist]
                # print(model.w.data)
                # print(f'State: {state}, Action: {action}')
                # print(f'Greedy action: {self.policy(model, state, 0)}')

            # update steps
            total_reward += reward
            self.num_steps += 1

        # get info, stored as dict
        info = self.episode_info(total_reward, action, reward, termination_flag)
        return info