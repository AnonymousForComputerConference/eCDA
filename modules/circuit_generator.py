#%%
import numpy as np
import networkx as nx
import random
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()
import matplotlib.pyplot as plt

from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory
from PySpice.Unit import *
from PySpice.Plot.BodeDiagram import bode_diagram_gain

import pickle

import dgl
from dgl import DGLGraph

ELEMENTS = ['R', 'C', 'L']

def add_resistor(circuit, r_value = None, random = False):
    if random:
        r_value = np.random.rand

def get_unit(element_type):
    t = element_type
    if t == 'R':
        return kilo
    if t == 'C':
        return micro
    if t == 'L':
        return micro

def plot_circuit_graph(g):
    print(len(g.edges))
    plt.subplot(111)
    nx.draw(g, with_labels=True)
    plt.show()

def plot_gains(feq, gains, titles, save_path = None):
    num_plot = len(gains)
    plt.close()
    figure = plt.figure(1, (10, 8))
    plt.title(titles[-1])
    for ix, gain in enumerate(gains):
        ax1 = plt.subplot(num_plot, 1, ix + 1)
        bode_diagram_gain(ax1,
                    frequency=feq,
                    gain=gain,
                    marker='.',
                    color='blue',
                    linestyle='-',
                )
        ax1.set_title(titles[ix])
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def plot_gains_reinforce(feq, gains, titles, background, save_path = None):
    num_plot = len(gains)
    plt.close()
    figure = plt.figure(1, (10, 4 * len(gains)))
    plt.title(titles[-1])
    for ix, gain in enumerate(gains):
        ax1 = plt.subplot(num_plot, 1, ix + 1)
        bode_diagram_gain(ax1,
                    frequency=feq,
                    gain=gain,
                    marker='.',
                    color='blue',
                    linestyle='-',
                )
        bode_diagram_gain(ax1,
                    frequency=feq,
                    gain=background,
                    # marker='.',
                    color='red',
                    linestyle='--',
                    alpha=0.5,
                )
        ax1.set_title(titles[ix])
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def circuit_graph_parser(circuit_file: str):
    """
    Read circuit from local text
    Return: nx.Graph()
    """
    g = nx.Graph()



def circuit_graph_generator(num_element: int, num_node = None, element_type = ELEMENTS):
    """
        Generate circuit graph with given number of nodes and edges.
        Edges are elements in the circuit and nodes are connections.
        0 - 1 is the voltage source
        In very rear cases, there can be triangle
        Return: nx.Graph()
    """
    if num_node is None:
        num_node = np.random.randint(np.ceil(np.sqrt(num_element * 2)), num_element + 1)
    num_edge = num_element
    g = nx.Graph()
    t_onehot = dict(zip(element_type, np.eye(len(element_type), dtype=np.float32)))
    circle_size = np.random.randint(3, num_node + 1)
    # add circle to the graph, (0,1) will be the voltage source so don't add it
    edges = [(i, i+1) for i in range(circle_size - 1)]
    edges.append((circle_size - 1, 0))
    for e in edges:
        t = random.choice(element_type)
        g.add_edge(e[0], e[1], t=t, v = np.random.choice(10)+np.array([1.0], dtype=np.float32), t1h = t_onehot[t])
    # for remained nodes, randomly add them to the circle
    remained_nodes = list(range(circle_size, num_node))
    num_remained_edge = num_edge - circle_size
    for node in remained_nodes:
        cands = [nd for nd in g.nodes if g.degree[nd] == 1]
        if len(cands) == 0:
            cands = g.nodes
        else:
            cands = list(cands)
            cands.extend(cands)
            cands.extend(cands)
            cands.extend(list(g.nodes))
        begin = np.random.choice(cands)
        g.add_node(node)
        t = random.choice(element_type)
        g.add_edge(begin, node, t=t, v = np.random.choice(10)+np.array([1.0], dtype=np.float32), t1h = t_onehot[t])
        num_remained_edge -= 1
    # for remained edges, solve the one-degree nodes' problem
    for _ in range(num_remained_edge):
        cands = [nd for nd in g.nodes if g.degree[nd] == 1]
        if len(cands) == 0:
            cands = g.nodes
        left = np.random.choice(cands)
        remained = list(g.nodes)
        remained.remove(left)
        for nd in list(g.neighbors(left)):
            remained.remove(nd)
        if len(remained) > 0:
            right = np.random.choice(remained)
            t = random.choice(element_type)
            g.add_edge(left, right, t=t, v = np.random.choice(10)+np.array([1.0], dtype=np.float32), t1h = t_onehot[t])
    # check degree = 1 nodes, move them to father nodes
    for node in g.nodes:
        if g.degree[node] > 1:
            continue
        nbhd = list(g.neighbors(node))[0]
        last_nbhd = node
        while g.degree[nbhd] <= 2:
            nbhds = list(g.neighbors(nbhd))
            nbhds.remove(last_nbhd)
            last_nbhd = nbhd
            nbhd = nbhds[0]
        cands = list(g.neighbors(nbhd))
        cands.remove(last_nbhd)
        if 0 in cands and nbhd == 1:
            cands.remove(0)
        if 1 in cands and nbhd == 0:
            cands.remove(1)
        selected = np.random.choice(cands)
        g.remove_edge(selected, nbhd)
        t = random.choice(element_type)
        g.add_edge(selected, node, t=t, v = np.random.choice(10)+np.array([1.0], dtype=np.float32), t1h = t_onehot[t])
    
    g.remove_edge(0, 1)
    return g

def graph_to_circuit(g, name = 'test', element_type = ELEMENTS, ac_amp = 1):
    c = Circuit(name)
    c.SinusoidalVoltageSource('ac', 1, 0, amplitude=ac_amp)
    name_cnt = dict.fromkeys(element_type, 1)
    for edge in g.edges:
        t = g.edges[edge]['t']
        v = g.edges[edge]['v']
        adding_method = getattr(c, t)
        if t == 'R':
            unit = kilo
        if t == 'C':
            unit = micro
        
        adding_method(name_cnt[t], edge[0], edge[1], unit(v))
        name_cnt[t] += 1
    return c

def random_modify(g, budget = 3, element_type = ELEMENTS, copy = True):
    # generate target circuit graph
    if copy:
        g = g.copy()
    t_onehot = dict(zip(element_type, np.eye(len(element_type), dtype=int)))
    can_modify = list(g.edges)
    log = []
    for _ in range(budget):
        action = random.choice([0,1]) # modify or add
        if action == 0:
            # modify
            selected = random.choice(can_modify)
            can_modify.remove(selected)
            g.remove_edge(*selected)
            t = random.choice(element_type)
            old_t = g.edges[(selected[0], selected[1])]['t']
            old_v = g.edges[(selected[0], selected[1])]['v']
            new_v = np.random.choice(10)+np.array([1.0], dtype=np.float32)
            g.add_edge(selected[0], selected[1], t = t, v=new_v, t1h = t_onehot[t])
            log.append(f'Change {selected} from {old_t}({old_v}) to {t}({new_v})')
        if action == 1:
            node1 = np.random.choice(g.nodes)
            remained = list(g.nodes)
            remained.remove(node1)
            node2 = np.random.choice(remained)
            t = random.choice(element_type)
            new_v = np.random.choice(10)+np.array([1.0], dtype=np.float32)
            g.add_edge(node1, node2, t = t, v=new_v, t1h = t_onehot[t])
            log.append(f'Add ({node1}, {node2}) as {t}({new_v})')
    return g, log

def simulate_circuit_graph(circuit_graph, name, s_feq = 1@u_Hz, e_feq = 1@u_kHz, number_of_points = 10, temp = 25, norm =False):
    """
    Input a circuit graph, output its frequencies of each node
    Return analysis.frequency
    """
    circuit = graph_to_circuit(circuit_graph, name)
    simulator = circuit.simulator(temperature=temp, nominal_temperature=temp)
    analysis = simulator.ac(start_frequency=s_feq, stop_frequency=e_feq, number_of_points=number_of_points,  variation='dec')

    for node in range(2, circuit_graph.number_of_nodes()):
        select_analysis = analysis[str(node)]
        gain = 20*np.log10(np.absolute(select_analysis))
        if norm:
            # normalize to 0-1
            gain = (gain - np.min(gain)) / (np.max(gain) - np.min(gain))
        circuit_graph.nodes[node]['gain'] = gain
    circuit_graph.nodes[0]['gain'] = circuit_graph.nodes[1]['gain'] = np.zeros_like(circuit_graph.nodes[2]['gain'])
    
    return analysis.frequency, circuit_graph

def simulate_circuit(circuit, node_num, s_feq = 1@u_Hz, e_feq = 1@u_kHz, number_of_points = 10, temp = 25, norm =True):
    simulator = circuit.simulator(temperature=temp, nominal_temperature=temp)
    analysis = simulator.ac(start_frequency=s_feq, stop_frequency=e_feq, number_of_points=number_of_points, variation='dec')
    gains = []
    for node in range(2, node_num):
        select_analysis = analysis[str(node)]
        gain = 20*np.log10(np.absolute(select_analysis))
        if norm:
            # normalize to 0-1
            gain = (gain - np.min(gain)) / (np.max(gain) - np.min(gain))
        gains.append(gain)
    gains = [np.zeros_like(gains[0]), np.zeros_like(gains[0])] + gains

    return gains


class CircuitDataset():
    def __init__(self, num_circuit = 50, budget = 3, save_path = '../data/circuits.pkl'):
        self.num_circuit = num_circuit
        circuits = []
        try:
            circuits, feq = pickle.load(open(save_path, 'rb'))
        except:
            pass
        if len(circuits) < self.num_circuit:
            remained_num = self.num_circuit - len(circuits)
            i = 0
            while i < remained_num:
                try:
                    source = circuit_graph_generator(10+random.choice([-1,0,1]), 6)
                    target, log = random_modify(source, budget=budget)
                    feq, source = simulate(source, f'c{i}')
                    _, target = simulate(target, f'tc{i}')
                except:
                    print('Circuit failed for simulation')
                    continue
                i+=1
                circuits.append((source, target, log))
            pickle.dump((circuits, feq), open(save_path, 'wb'))
        
            # # convert to dgl data
            # e_type_map = {
            #     'R': np.array([1,0]),
            #     'C': np.array([0,1])
            # }

            # dgl_circuit = []
            # for org, tgt, log in circuits:
            #     dgl_org = DGLGraph()
            #     dgl_org.from_networkx(org, node_attrs=['gain'])
            #     for e in org.edges:
            #         e_ftr = e_type_map[org.edges[e]['t']]

        self.feq = feq
        self.circuits = circuits




#%%
if __name__ == '__main__':

    from PySpice.Plot.BodeDiagram import bode_diagram_gain
    plt.title("Test")
    figure = plt.figure(1, (20, 10))
    axe = plt.subplot(111)
    
    bode_diagram_gain(axe,
                frequency=analysis.frequency,
                gain=gains[5],
                marker='.',
                color='blue',
                linestyle='-',
            )

    plt.tight_layout()
    plt.savefig('gain_example.pdf')