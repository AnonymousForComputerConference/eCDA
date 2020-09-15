import numpy as np
import networkx as nx
import random
from copy import copy
import math
from collections import deque
import time
import os

import torch
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim

from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *

from modules.circuit_generator import *
from modules.graph import CircuitConv

def my_init(layers):
    for layer in layers:
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)


def diff(gain0, gain1):
    return th.mean((gain0 - gain1)**2)

def normalized_columns_initializer_tf(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out

class Environment(object):
    def __init__(self, circuit_graph, target_circuit_graph, max_step, target_node, element_type = ['R', 'C']):
        self.nx_source = circuit_graph
        self.nx_target = target_circuit_graph
        # self.source = DGLGraph()
        # self.source.from_networkx(self.nx_source, node_attrs=['gain'], edge_attrs=['t1h','v'])
        # self.target = DGLGraph()
        # self.target.from_networkx(self.nx_target, node_attrs=['gain'], edge_attrs=['t1h','v'])


        # compute gain
        self.pick_out_node = target_node #np.random.choice(np.arange(2, self.nx_target.number_of_nodes()))
        self.target_gain = th.from_numpy(self.nx_target.nodes[self.pick_out_node]['gain'][np.newaxis])
        self.source_gain = th.from_numpy(self.nx_source.nodes[self.pick_out_node]['gain'][np.newaxis])
        self.gain_diff = diff(self.target_gain, self.source_gain)
        self.t_onehot = dict(zip(element_type, th.eye(len(element_type), dtype=th.float32)))
        self.element_type = element_type
        self.max_step = max_step

        self.reset()

    def normalized_reward(self, current_gain):
        _diff = diff(self.target_gain, current_gain)
        n_r = 1.0 - _diff / self.gain_diff
        if n_r < 0:
            n_r = n_r - n_r
        return n_r

    def reset(self):
        # dgl graph
        # self._circuit_graph = copy(self.source)
        self.n_step = 0
        self._circuit_graph = DGLGraph()
        self._circuit_graph.from_networkx(self.nx_source, node_attrs=['gain'], edge_attrs=['t1h','v'])
        # pyspice circuit
        self._circuit = graph_to_circuit(self.nx_source)
        self.name_cnt = dict.fromkeys(self.element_type, 1)
        return self._circuit_graph

    def step(self, action):
        # action (n1, n2, t, v)
        self.n_step += 1
        n1, n2, t, v = action
        n1 = n1[0][0].numpy()
        n2 = n2[0][0].numpy()
        t = self.element_type[t[0].numpy()]
        v = v.detach()

        # update circuit
        adding_method = getattr(self._circuit, t)
        if t == 'R':
            unit = kilo
        if t == 'C':
            unit = micro
        adding_method(f'add{self.name_cnt[t]}', n1, n2, unit(v))
        self.name_cnt[t] += 1
        gains = simulate_circuit(self._circuit, self._circuit_graph.number_of_nodes())
        gains = th.from_numpy(np.array(gains))
        # update circuit graph
        t1h = self.t_onehot[t][np.newaxis]
        self._circuit_graph.add_edge(n1, n2, {'v': v, 't1h': t1h})
        self._circuit_graph.add_edge(n2, n1, {'v': v, 't1h': t1h})
        self._circuit_graph.ndata['gain'] = gains
        # error = diff(self.target_gain, self._circuit_graph.nodes[self.pick_out_node].data['gain'])
        # errors = [diff(self.target, gains[i]) for i in range(2, self._circuit.number_of_nodes)]
        # min_idx = np.argmin(errors)
        # error = np.min(errors)
        current_gain = self._circuit_graph.nodes[self.pick_out_node].data['gain']
        reward = self.normalized_reward(current_gain)

        return self._circuit_graph, reward, self.n_step >= self.max_step, diff(self.target_gain, current_gain)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class AMetaCDANetCNet(nn.Module):
    # def __init__(self, num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = 64):
    def __init__(self, num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = 64):
        super().__init__()
        self.node_ftr_dim = node_ftr_dim
        self.edge_ftr_dim = edge_ftr_dim
        self.a1_size = a1_size
        self.a2_size = a2_size
        self.hidden_dim = hidden_dim
        self.cconv = CircuitConv(node_ftr_dim + edge_ftr_dim, hidden_dim)

        flatten_dim = hidden_dim * num_node #+ a1_size + a2_size + num_type + 1 + 1
        self.lstm = nn.LSTMCell(flatten_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, 1) # estimate the value
        self.a1_linear = nn.Linear(hidden_dim, a1_size) # node 1
        self.a2_linear = nn.Linear(hidden_dim + a1_size, a2_size) # node 2
        self.type_linear = nn.Linear(hidden_dim + a1_size + a2_size, num_type) # select element type
        self.ev_linear = nn.Linear(hidden_dim + a1_size + a2_size + num_type, 1) # value for element
        self.distribution = torch.distributions.Categorical

        self.apply(weights_init)
        self.a1_linear.weight.data = normalized_columns_initializer(
            self.a1_linear.weight.data, 0.01)
        self.a1_linear.bias.data.fill_(0)
        self.a2_linear.weight.data = normalized_columns_initializer(
            self.a2_linear.weight.data, 0.01)
        self.a2_linear.bias.data.fill_(0)
        self.value_linear.weight.data = normalized_columns_initializer(
            self.value_linear.weight.data, 1.0)
        self.value_linear.bias.data.fill_(0)


        self.type_linear.weight.data = normalized_columns_initializer(
            self.type_linear.weight.data, 0.01)
        self.type_linear.bias.data.fill_(0)
        self.ev_linear.weight.data = normalized_columns_initializer(
            self.ev_linear.weight.data, 0.01)
        self.ev_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)


    def forward(self, circuits, h0, c0):
        if type(circuits) == list:
            x = [th.flatten(self.cconv(circuit)) for circuit in circuits]
        else:
            x = [th.flatten(self.cconv(circuits)),]
        x = th.stack(x, dim = 0)
        hn, cn = self.lstm(x, (h0, c0))
        x = hn
        values = self.value_linear(x)
        a1s = self.a1_linear(x)
        x = th.cat([x, a1s], dim = -1)
        a2s = self.a2_linear(x)
        x = th.cat([x, a2s], dim = -1)
        types = self.type_linear(x)
        x = th.cat([x, types], dim = -1)
        element_values = self.ev_linear(x)
        return values, (a1s, a2s, types, element_values), (hn, cn)


class MetaACNet(nn.Module):
    # def __init__(self, num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = 64):
    def __init__(self, num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = 64):
        super().__init__()
        self.node_ftr_dim = node_ftr_dim
        self.edge_ftr_dim = edge_ftr_dim
        self.a1_size = a1_size
        self.a2_size = a2_size
        self.hidden_dim = hidden_dim
        self.cconv = CircuitConv(node_ftr_dim + edge_ftr_dim, hidden_dim)

        flatten_dim = hidden_dim * num_node + a1_size + a2_size + num_type + 1 + 1
        self.lstm = nn.LSTM(flatten_dim, hidden_dim)
        self.value_linear = nn.Linear(hidden_dim, 1) # estimate the value
        self.a1_linear = nn.Linear(hidden_dim, a1_size) # node 1
        self.a2_linear = nn.Linear(hidden_dim + a1_size, a2_size) # node 2
        self.type_linear = nn.Linear(hidden_dim + a1_size + a2_size, num_type) # select element type
        self.ev_linear = nn.Linear(hidden_dim + a1_size + a2_size + num_type, 1) # value for element
        self.distribution = torch.distributions.Categorical

        my_init([self.value_linear, self.a1_linear, self.a2_linear, self.type_linear, self.ev_linear])

        # todo: init for lstm

    def forward(self, circuits, prev_rewards, prev_a1s, prev_a2s, prev_types, prev_evs, time_steps, h0, c0):
        if type(circuits) == list:
            x = [th.flatten(self.cconv(circuit)) for circuit in circuits]
        else:
            x = [th.flatten(self.cconv(circuits)),]
        x = th.stack(x, dim = 0)
        x = th.cat([x, prev_rewards, prev_a1s, prev_a2s, prev_types, prev_evs, time_steps], dim = -1)
        x, (hn, cn) = self.lstm(x, (h0, c0))
        values = self.value_linear(x)
        a1s = self.a1_linear(x)
        x = th.cat([x, a1s], dim = -1)
        a2s = self.a2_linear(x)
        x = th.cat([x, a2s], dim = -1)
        types = self.type_linear(x)
        x = th.cat([x, types], dim = -1)
        element_values = self.ev_linear(x)
        return values, (a1s, a2s, th.argmax(types), element_values), (hn, cn)

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

def train(rank, envarg, args, shared_model, counter, lock, optimizer=None):
    torch.manual_seed(args.seed + rank)
    env = Environment(*envarg)
    # env = create_atari_env(args.env_name)
    # env.seed(args.seed + rank)

    num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = args.num_node, args.node_ftr_dim, args.edge_ftr_dim, args.a1_size, args.a2_size, args.num_type, args.hidden_dim

    model = AMetaCDANetCNet(num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()

    state = env.reset()
    # state = torch.from_numpy(state)
    done = True
    # cx = torch.zeros(1, args.hidden_dim)
    # hx = torch.zeros(1, args.hidden_dim)

    episode_length = 0
    trained_time = 0
    while True:
        trained_time += 1
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, args.hidden_dim)
            hx = torch.zeros(1, args.hidden_dim)
        else:
            cx = cx.detach()
            hx = hx.detach()
        # cx = cx.detach()
        # hx = hx.detach()

        values = []
        log_probs = []
        rewards = []
        entropies = []

        for step in range(args.num_steps):
            episode_length += 1
            value, (a1s, a2s, types, element_values), (hx, cx) = model(state,
                                            hx, cx)

            prob_a1 = F.softmax(a1s, dim=-1)
            log_prob_a1 = F.log_softmax(a1s, dim=-1)

            prob_a2 = F.softmax(a2s, dim=-1)
            log_prob_a2 = F.log_softmax(a2s, dim=-1)

            entropy_a1 = -(prob_a1 * log_prob_a1).sum(1, keepdim=True)
            entropy_a2 = -(prob_a2 * log_prob_a2).sum(1, keepdim=True)
            entropies.append(entropy_a1 + entropy_a2)

            action_a1 = prob_a1.multinomial(num_samples=1).detach()
            action_a2 = prob_a2.multinomial(num_samples=1).detach()
            log_prob_a1 = log_prob_a1.gather(1, action_a1)
            log_prob_a2 = log_prob_a2.gather(1, action_a2)

            types = th.argmax(types, dim=-1)
            element_values = th.sigmoid(element_values) * 9 + 1.0

            state, reward, done, _ = env.step((action_a1, action_a2, types, element_values))
            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                episode_length = 0
                state = env.reset()

            # state = torch.from_numpy(state)
            values.append(value)
            log_probs.append((log_prob_a1 + log_prob_a2) / 2.0)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model(state, hx, cx)
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        # print(f'Rewards: {rewards}')
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimation
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        with lock:
            if counter.value > args.max_counter_value:
                break
    print(f'Worker {rank} finished after simulating {trained_time} times.')


def test(rank, envarg, args, shared_model, counter, lock):
    torch.manual_seed(args.seed + rank)
    logger = open(os.path.join(args.circuit_path, 'testing_log_tmp.txt'), 'a')

    env = Environment(*envarg)

    num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = args.num_node, args.node_ftr_dim, args.edge_ftr_dim, args.a1_size, args.a2_size, args.num_type, args.hidden_dim

    model = AMetaCDANetCNet(num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim)

    model.eval()

    state = env.reset()
    # state = torch.from_numpy(state)
    best_reward_sum = 0
    reward_sum = 0
    gains = []
    done = True
    historical_rewards = []

    start_time = time.time()

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=100)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = torch.zeros(1, args.hidden_dim)
            hx = torch.zeros(1, args.hidden_dim)
        else:
            cx = cx.detach()
            hx = hx.detach()

        with torch.no_grad():
            value, (a1s, a2s, types, element_values), (hx, cx) = model(state,
                                            hx, cx)
            prob_a1 = F.softmax(a1s, dim=-1)

            prob_a2 = F.softmax(a2s, dim=-1)

            action_a1 = prob_a1.max(1, keepdim=True)[1]
            action_a2 = prob_a2.max(1, keepdim=True)[1]

            types = th.argmax(types, dim=-1)
            element_values = th.sigmoid(element_values) * 9 + 1.0

        state, reward, done, gain_diff = env.step((action_a1, action_a2, types, element_values))
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward
        gains.append(gain_diff.cpu().numpy()[()])

        # a quick hack to prevent the agent from stucking
        actions.append((action_a1, action_a2, types, element_values))
        # if actions.count((action_a1, action_a2)) == actions.maxlen:
        #     done = True

        if done:
            historical_rewards.append((counter.value, reward_sum.cpu().numpy()[()]))
            log_str = "Time {}, num steps {}, FPS {:.0f}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                counter.value, counter.value / (time.time() - start_time),
                reward_sum, episode_length)
            action_strs = [f'{args.element_type[action[2][0].numpy()]}({str(np.round(action[3][0][0].numpy(), 4))}) between ({action[0][0][0].numpy()}, {action[1][0][0].numpy()})' for action in actions]
            # if args.verbose:
            #     print(log_str)
            logger.write(log_str + '\n')
            logger.write('Actions:\n')
            logger.write('\n'.join(action_strs))
            logger.write('\n')
            logger.flush()
            
            
            # if improved, plot the design process and save the model
            if reward_sum > best_reward_sum:
                print(f'Reward improved to {reward_sum}!!!')
                print(gains, np.mean(gains))
                best_reward_sum = reward_sum
                # titles = [f'Step {i}: {action_strs[i]}' for i in range(len(gains))]
                # titles.append('Design Process')
                # plot_gains_reinforce(args.feq, gains, titles, args.target_gain, os.path.join(args.circuit_path, 'design_process.pdf'))
                # torch.save(shared_model, open(os.path.join(args.circuit_path, 'saved_model_tmp.pkl'), 'wb'))

            reward_sum = 0
            episode_length = 0
            gains = []
            actions.clear()
            state = env.reset()
            time.sleep(0.01)

        with lock:
            if counter.value > args.max_counter_value:
                pickle.dump(historical_rewards, open(os.path.join(args.circuit_path, 'historical_rewards.pkl'), 'wb'))
                break
    
    logger.close()