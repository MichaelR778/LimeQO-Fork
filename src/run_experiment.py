import argparse
import torch
from tqdm import tqdm
from data.dataset import Dataset
from strategies.oracle import OracleStrategy
from strategies.qo_advisor import QOAdvisorStrategy
from strategies.random import RandomStrategy
from strategies.greedy import GreedyStrategy
from strategies.limeqo import LimeQOStrategy
from strategies.limeqo_plus import LimeQOPlusStrategy
from strategies.another_greedy import AnotherGreedyStrategy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ceb', required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--rank', type=int, default=5)
    parser.add_argument('--modified', type=bool, default=False)
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
    dataset = Dataset(dataset_name)

    max_duration = 0
    if args.modified:
        print("MODIFIED VER ON")
        if dataset_name == "job":
            max_duration = 400
        elif dataset_name == "ceb":
            max_duration = 21600
    
    # Oracle
    # print("Running oracle")
    # oracle = OracleStrategy()
    # oracle.run(dataset, f"experiment/{args.dataset}/oracle.json")
    
    # QO Advisor
    # print("Running QO Advisor")
    # qo_advisor = QOAdvisorStrategy()
    # qo_advisor.run(dataset, f"experiment/{args.dataset}/qo_advisor/1.json")
    
    # Random
    print("Running random")
    random = RandomStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        random.run(dataset, f"experiment/{args.dataset}/random/{i}.json", max_duration)
    
    # # Greedy
    print("Running greedy")
    greedy = GreedyStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        greedy.run(dataset, f"experiment/{args.dataset}/greedy/{i}.json", max_duration)
    
    # LimeQO
    print("Running limeqo")
    limeqo = LimeQOStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        limeqo.run(dataset, f"experiment/{args.dataset}/limeqo/{i}.json", max_duration)
    
    # LimeQO+
    print("Running limeqo+")
    limeqo_plus = LimeQOPlusStrategy(new_observe_size=32, device=device)
    if dataset_name == "ceb":
        # run once
        limeqo_plus.run(dataset, f"experiment/{args.dataset}/limeqo+/1.json", max_duration) #run once only for experimental purpose
    else:
        for i in tqdm(range(1, 6)):
            limeqo_plus.run(dataset, f"experiment/{args.dataset}/limeqo+/{i}.json", max_duration)

    # Another Greedy
    print("Running another greedy")
    another_greedy = AnotherGreedyStrategy(new_observe_size=8)
    for i in tqdm(range(1, 21)):
        another_greedy.run(dataset, f"experiment/{args.dataset}/another_greedy/{i}.json", max_duration, 0.5)

def main():
    args = parse_args()
    device = setup_device(args)
    run_experiments(args, device)

if __name__ == '__main__':
    main() 