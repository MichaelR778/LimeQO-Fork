import argparse
import torch
from tqdm import tqdm
from data.dataset import Dataset
from data.partial_dataset import PartialDataset
from strategies.greedyshift import GreedyshiftStrategy
from strategies.limeqoshift import LimeQOshiftStrategy
from strategies.another_greedyshift import AnotherGreedyshiftStrategy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ceb', required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rank', type=int, default=5)
    return parser.parse_args()

def setup_device(args):
    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    return device

def run_experiments(args, device):
    """Run all experimental strategies"""
    dataset_name = args.dataset
    partial_dataset = PartialDataset(dataset_name, 0.7)
    full_dataset = Dataset(dataset_name)

    if dataset_name == "job":
        max_duration = 400
    elif dataset_name == "ceb":
        max_duration = 21600

    # Greedy with workload shift
    print("Running greedy with workload shift")
    greedyshift = GreedyshiftStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        greedyshift.run(partial_dataset, full_dataset, f"experiment/{dataset_name}/newquery/greedy_newquery_timeout_run{i}.json", max_duration)
    
    # LimeQO with workload shift
    print("Running limeqo with workload shift")
    limeqoshift = LimeQOshiftStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        limeqoshift.run(partial_dataset, full_dataset, f"experiment/{dataset_name}/newquery/als_newquery_timeout_rank5_lambda0.2_alpha1_beta15_run{i}.json", max_duration)

    # Another Greedy with workload shift
    print("Running another greedy with workload shift")
    another_greedyshift = AnotherGreedyshiftStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        another_greedyshift.run(partial_dataset, full_dataset, f"experiment/{dataset_name}/newquery/another_greedy_newquery_timeout_run{i}.json", max_duration, 0.5)

def main():
    args = parse_args()
    device = setup_device(args)
    run_experiments(args, device)

if __name__ == '__main__':
    main() 