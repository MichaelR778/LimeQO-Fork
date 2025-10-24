import json
import os
import numpy as np
from strategies.base import BaseStrategy

class AnotherGreedyshiftStrategy(BaseStrategy):
    """Another greedy hint selection strategy with timeout for workload shift experiment"""
    
    def __init__(self, new_observe_size=32):
        self.new_observe_size = new_observe_size
        
    def run(self, partial_dataset, full_dataset, output_path, max_duration, explore_threshold = 0.5):
        """
        Run another greedy strategy with timeout
        
        Args:
            partial_dataset: Partial dataset object containing query data
            full_dataset: Full dataset object containing query data
            output_path: Path to save results
            max_duration: Max duration for offline exploration
            explore_threshold: 0.5 means explore only 50% of hints available
        """
        mask = np.zeros_like(partial_dataset.init_mask)
        for i in range(partial_dataset.matrix.shape[0]):
            same_hints = partial_dataset.get_same_hints(i, 0)
            mask[i, same_hints] = 1
            
        exec_time = partial_dataset.get_exec_time(mask)
        timeout_m = np.zeros(partial_dataset.matrix.shape)
        min_observed = partial_dataset.get_min_observed(partial_dataset.matrix, mask)
        timeout = 0
        results = []
        explore_queries = set()

        while exec_time < ((max_duration / 3) + partial_dataset.default_time):
            exec_time = partial_dataset.get_exec_time(mask) + timeout
            min_observed = partial_dataset.get_min_observed(partial_dataset.matrix, mask)
            
            results.append({
                "training_time": 0,
                "inference_time": 0,
                "exec_time": exec_time,
                "total_latency": np.sum(min_observed),
                "p50": np.median(min_observed),
                "p90": np.percentile(min_observed, 90),
                "p95": np.percentile(min_observed, 95),
                "p99": np.percentile(min_observed, 99),
                "explore_queries_cnt": len(explore_queries)
            })
            
            cnt = 0
            selects = np.argsort(-min_observed)
            
            for i in range(len(selects)):
                if cnt >= self.new_observe_size:
                    break
                    
                file_i = selects[i]
                # if mask[file_i].sum() == mask.shape[1]:
                #     continue
                mask_timeout_combined = np.maximum(mask[file_i], timeout_m[file_i])
                if (np.sum(mask_timeout_combined) / len(mask_timeout_combined)) >= explore_threshold:
                    continue
                    
                while True:
                    hint_i = np.random.randint(partial_dataset.matrix.shape[1])
                    if mask[file_i, hint_i] == 0:
                        if timeout_m[file_i, hint_i] == 1:
                            continue
                            
                        same_hints = partial_dataset.get_same_hints(file_i, hint_i)
                        
                        if partial_dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                            timeout_m[file_i, same_hints] = 1
                            timeout += min_observed[file_i]
                            break
                            
                        mask[file_i, same_hints] = 1
                        cnt += 1
                        explore_queries.add(file_i)
                        break

        # transfer required matrix from previous loop
        oldmask = mask
        mask = np.zeros_like(full_dataset.init_mask)
        for i in range(full_dataset.matrix.shape[0]):
            same_hints = full_dataset.get_same_hints(i, 0)
            mask[i, same_hints] = 1
        oldtimeout_m = timeout_m
        timeout_m = np.zeros(full_dataset.matrix.shape)
        # transfer
        for old_i, i in enumerate(partial_dataset.included_indexes):
            mask[i] = np.maximum(mask[i], oldmask[old_i])
            timeout_m[i] = np.maximum(timeout_m[i], oldtimeout_m[old_i])
        # transfer

        # loop on full dataset
        while exec_time < (max_duration + full_dataset.default_time):
            exec_time = full_dataset.get_exec_time(mask) + timeout
            min_observed = full_dataset.get_min_observed(full_dataset.matrix, mask)
            
            results.append({
                "training_time": 0,
                "inference_time": 0,
                "exec_time": exec_time,
                "total_latency": np.sum(min_observed),
                "p50": np.median(min_observed),
                "p90": np.percentile(min_observed, 90),
                "p95": np.percentile(min_observed, 95),
                "p99": np.percentile(min_observed, 99),
                "explore_queries_cnt": len(explore_queries)
            })
            
            cnt = 0
            selects = np.argsort(-min_observed)
            
            for i in range(len(selects)):
                if cnt >= self.new_observe_size:
                    break
                    
                file_i = selects[i]
                # if mask[file_i].sum() == mask.shape[1]:
                #     continue
                mask_timeout_combined = np.maximum(mask[file_i], timeout_m[file_i])
                if (np.sum(mask_timeout_combined) / len(mask_timeout_combined)) >= explore_threshold:
                    continue
                    
                while True:
                    hint_i = np.random.randint(full_dataset.matrix.shape[1])
                    if mask[file_i, hint_i] == 0:
                        if timeout_m[file_i, hint_i] == 1:
                            continue
                            
                        same_hints = full_dataset.get_same_hints(file_i, hint_i)
                        
                        if full_dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                            timeout_m[file_i, same_hints] = 1
                            timeout += min_observed[file_i]
                            break
                            
                        mask[file_i, same_hints] = 1
                        cnt += 1
                        explore_queries.add(file_i)
                        break
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)