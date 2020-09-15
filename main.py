import warnings
warnings.filterwarnings("ignore")
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging(logging_level='CRITICAL')
import argparse
import os
import sys
import torch
import torch.multiprocessing as mp


from modules.reinforce import MetaCDANet, train, test, Environment, SharedAdam
from modules.circuit_generator import *
from PySpice.Plot.BodeDiagram import bode_diagram_gain
# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='lambda parameter for GAE (default: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='value loss coefficient (default: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='value loss coefficient (default: 50)')
parser.add_argument('--seed', type=int, default=2,
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=15,
                    help='how many training processes to use (default: 15)')
parser.add_argument('--num-steps', type=int, default=3,
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000000,
                    help='maximum length of an episode (default: 10000000)')
parser.add_argument('--max-counter-value', type=int, default=50000,
                    help='maximum value of the counter (default: 500000)')
parser.add_argument('--env-name', default='A3C Circuit Design',
                    help='environment to train on (default: A3C Circuit Design)')
parser.add_argument('--no-shared', default=False,
                    help='use an optimizer without shared momentum.')

num_node = 6
element_type = ['R', 'C']
parser.add_argument('--num_node', type=int, default=num_node,
                    help='number of nodes/wires in the circuit')
parser.add_argument('--node_ftr_dim', type=int, default=31,
                    help='node feature dim')
parser.add_argument('--edge_ftr_dim', type=int, default=3,
                    help='edge feature dim (onehot type + value)')
parser.add_argument('--a1_size', type=int, default=num_node,
                    help='action size for node 1, equals to number of nodes')
parser.add_argument('--a2_size', type=int, default=num_node,
                    help='action size for node 2, equals to number of nodes')
parser.add_argument('--num_type', type=int, default=len(element_type),
                    help='number of elements')
parser.add_argument('--element_type', type=list, default=element_type,
                    help='elements in the circuit')
parser.add_argument('--hidden_dim', type=int, default=64,
                    help='hidden dim')

# path related
parser.add_argument('--save_path', type=str, default='../logs',
                    help='path to save the results')
parser.add_argument('--is_save_model', type=bool, default=True,
                    help='whether the test() saves the model or not')
parser.add_argument('--verbose', type=bool, default=True,
                    help='if output testing results on screen')



if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    circuit_dataset = CircuitDataset(50)
    args.circuit_dataset = circuit_dataset
    for ix in range(len(circuit_dataset.circuits)):
        circuit_path = os.path.join(args.save_path, f'{ix}')
        if not os.path.isdir(circuit_path):
            os.mkdir(circuit_path)
        args.circuit_path = circuit_path
        args.circuit_id = ix

        source = circuit_dataset.circuits[ix][0]
        target = circuit_dataset.circuits[ix][1]
        diff = circuit_dataset.circuits[ix][2]
        target_node = np.random.choice(np.arange(2, target.number_of_nodes()))

        # log some discriptions
        readme_path = os.path.join(circuit_path, 'readme.txt')
        if os.path.exists(readme_path):
            target_node = int(open(readme_path).readline().split(':')[-1])
        with open(readme_path, 'w') as readme:
            readme.write(f'Select node: {target_node}\n')
            readme.write('\nSource circuit:\n')
            readme.write(str(graph_to_circuit(source)).replace('\r', ''))
            readme.write('\nTarget circuit:\n')
            readme.write(str(graph_to_circuit(target)).replace('\r', ''))
            readme.write('\nModifying details:\n')
            readme.write('\n'.join(diff))
        

        # plot the gain curve
        feq = np.array(circuit_dataset.feq)
        args.feq = feq
        titles = ['Start', 'Target', 'Design Objective']
        gains = [source.nodes[target_node]['gain'].flatten(), target.nodes[target_node]['gain'].flatten()]
        args.target_gain = gains[-1]
        plot_gains_reinforce(feq, gains, titles, gains[0], os.path.join(circuit_path, 'design_obj.pdf'))




        num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim = args.num_node, args.node_ftr_dim, args.edge_ftr_dim, args.a1_size, args.a2_size, args.num_type, args.hidden_dim
        shared_model = MetaCDANet(num_node, node_ftr_dim, edge_ftr_dim, a1_size, a2_size, num_type, hidden_dim)
        shared_model.share_memory()

        if args.no_shared:
            optimizer = None
        else:
            optimizer = SharedAdam(shared_model.parameters(), lr=args.lr)
            optimizer.share_memory()

        processes = []

        counter = mp.Value('i', 0)
        lock = mp.Lock()

        p = mp.Process(target=test, args=(args.num_processes, (source, target, args.num_steps, target_node), args, shared_model, counter, lock))
        p.start()
        processes.append(p)
        
        # speed up with multiprocess
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, (source, target, args.num_steps, target_node), args, shared_model, counter, lock, optimizer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        
        print(f'Done for circuit {ix}.')
