import json
import os
import numpy as np
import time
from strategies.base import BaseStrategy
from models.matrix_factorization import censored_als

class LimeQOshiftStrategy(BaseStrategy):
    """LimeQO strategy using censored matrix factorization for workload shift experiment"""
    
    def __init__(self, rank=5, lambda_=0.2, alpha=1.0, beta=15.0, new_observe_size=32):
        self.rank = rank
        self.lambda_ = lambda_
        self.alpha = alpha
        self.beta = beta
        self.new_observe_size = new_observe_size
        
    def run(self, partial_dataset, full_dataset, output_path, max_duration):
        """
        Run LimeQO strategy
        
        Args:
            partial_dataset: Partial dataset object containing query data
            full_dataset: Full dataset object containing query data
            output_path: Path to save results
            max_duration: Max duration for offline exploration
        """
        mask = partial_dataset.init_mask.copy()
        exec_time = partial_dataset.get_exec_time(mask)
        timeout_m = np.zeros_like(partial_dataset.matrix)
        explored_m = partial_dataset.init_mask.copy()
        min_observed = partial_dataset.get_min_observed(partial_dataset.matrix, mask)
        timeout = 0
        results = []
        explore_queries = set()

        while exec_time < ((max_duration / 3) + partial_dataset.default_time):
            exec_time = partial_dataset.get_exec_time(mask) + timeout
            min_observed = partial_dataset.get_min_observed(partial_dataset.matrix, mask)
            
            masked_m = partial_dataset.matrix * mask
            log_m = np.log1p(masked_m)
            log_timeout_m = np.log1p(timeout_m)
            
            start_time = time.time()
            pred_m = censored_als(log_m, mask, log_timeout_m, self.rank, 50, self.lambda_)
            training_time = time.time() - start_time
            pred_m = np.expm1(pred_m)
            
            pred_m = pred_m * (1-mask)
            pred_m[pred_m == 0] = np.inf
            start_time = time.time()
            mc_select = np.argmin(pred_m, axis=1)
            inference_time = time.time() - start_time
            
            results.append({
                "training_time": training_time,
                "inference_time": inference_time,
                "exec_time": exec_time,
                "total_latency": np.sum(min_observed),
                "p50": np.median(min_observed),
                "p90": np.percentile(min_observed, 90),
                "p95": np.percentile(min_observed, 95),
                "p99": np.percentile(min_observed, 99),
                "explore_queries_cnt": len(explore_queries)
            })
            
            mc_min = np.min(pred_m, axis=1)
            improve = (min_observed - mc_min) / mc_min
            
            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= self.new_observe_size:
                    break
                hint = mc_select[select]
                timeout_tolerance = min(self.alpha * min_observed[select], 
                                     self.beta * pred_m[select, hint])
                
                if (np.isinf(pred_m[select, hint]) or 
                    explored_m[select, hint] != 0 or 
                    pred_m[select, hint] >= timeout_tolerance):
                    continue
                
                same_hints = partial_dataset.get_same_hints(select, hint)
                
                if partial_dataset.matrix[select, hint] >= min_observed[select]:
                    explored_m[select, same_hints] = 1
                
                if partial_dataset.matrix[select, hint] >= timeout_tolerance:
                    timeout_m[select, same_hints] = timeout_tolerance
                    timeout += timeout_tolerance
                    continue
                
                mask[select, same_hints] = 1
                explored_m[select, same_hints] = 1
                cnt += 1
                explore_queries.add(select)
            
            # Random exploration if needed
            while cnt < self.new_observe_size:
                min_observed = partial_dataset.get_min_observed(partial_dataset.matrix, mask)
                if min_observed.sum() <= partial_dataset.opt_time + 50:
                    break
                    
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                
                if mask[file_i, hint_i] == 0 and explored_m[file_i, hint_i] == 0:
                    same_hints = partial_dataset.get_same_hints(file_i, hint_i)
                    
                    if partial_dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        explored_m[file_i, same_hints] = 1
                        timeout_m[file_i, same_hints] = min_observed[file_i]
                        continue
                    
                    explored_m[file_i, same_hints] = 1
                    mask[file_i, same_hints] = 1
                    cnt += 1
                    explore_queries.add(file_i) 

        # transfer required matrix from previous loop
        oldmask = mask
        mask = full_dataset.init_mask.copy()
        oldtimeout_m = timeout_m
        timeout_m = np.zeros_like(full_dataset.matrix)
        oldexplored_m = explored_m
        explored_m = full_dataset.init_mask.copy()
        # transfer
        for old_i, i in enumerate(partial_dataset.included_indexes):
            mask[i] = np.maximum(mask[i], oldmask[old_i])
            timeout_m[i] = np.maximum(timeout_m[i], oldtimeout_m[old_i])
            explored_m[i] = np.maximum(explored_m[i], oldexplored_m[old_i])
        # transfer

        # loop on full dataset
        while exec_time < (max_duration + full_dataset.default_time):
            exec_time = full_dataset.get_exec_time(mask) + timeout
            min_observed = full_dataset.get_min_observed(full_dataset.matrix, mask)
            
            masked_m = full_dataset.matrix * mask
            log_m = np.log1p(masked_m)
            log_timeout_m = np.log1p(timeout_m)
            
            start_time = time.time()
            pred_m = censored_als(log_m, mask, log_timeout_m, self.rank, 50, self.lambda_)
            training_time = time.time() - start_time
            pred_m = np.expm1(pred_m)
            
            pred_m = pred_m * (1-mask)
            pred_m[pred_m == 0] = np.inf
            start_time = time.time()
            mc_select = np.argmin(pred_m, axis=1)
            inference_time = time.time() - start_time
            
            results.append({
                "training_time": training_time,
                "inference_time": inference_time,
                "exec_time": exec_time,
                "total_latency": np.sum(min_observed),
                "p50": np.median(min_observed),
                "p90": np.percentile(min_observed, 90),
                "p95": np.percentile(min_observed, 95),
                "p99": np.percentile(min_observed, 99),
                "explore_queries_cnt": len(explore_queries)
            })
            
            mc_min = np.min(pred_m, axis=1)
            improve = (min_observed - mc_min) / mc_min
            
            selects = np.argsort(-improve)
            cnt = 0
            for select in selects:
                if cnt >= self.new_observe_size:
                    break
                hint = mc_select[select]
                timeout_tolerance = min(self.alpha * min_observed[select], 
                                     self.beta * pred_m[select, hint])
                
                if (np.isinf(pred_m[select, hint]) or 
                    explored_m[select, hint] != 0 or 
                    pred_m[select, hint] >= timeout_tolerance):
                    continue
                
                same_hints = full_dataset.get_same_hints(select, hint)
                
                if full_dataset.matrix[select, hint] >= min_observed[select]:
                    explored_m[select, same_hints] = 1
                
                if full_dataset.matrix[select, hint] >= timeout_tolerance:
                    timeout_m[select, same_hints] = timeout_tolerance
                    timeout += timeout_tolerance
                    continue
                
                mask[select, same_hints] = 1
                explored_m[select, same_hints] = 1
                cnt += 1
                explore_queries.add(select)
            
            # Random exploration if needed
            while cnt < self.new_observe_size:
                min_observed = full_dataset.get_min_observed(full_dataset.matrix, mask)
                if min_observed.sum() <= full_dataset.opt_time + 50:
                    break
                    
                file_i = np.random.randint(mask.shape[0])
                hint_i = np.random.randint(mask.shape[1])
                
                if mask[file_i, hint_i] == 0 and explored_m[file_i, hint_i] == 0:
                    same_hints = full_dataset.get_same_hints(file_i, hint_i)
                    
                    if full_dataset.matrix[file_i, hint_i] >= min_observed[file_i]:
                        timeout += min_observed[file_i]
                        explored_m[file_i, same_hints] = 1
                        timeout_m[file_i, same_hints] = min_observed[file_i]
                        continue
                    
                    explored_m[file_i, same_hints] = 1
                    mask[file_i, same_hints] = 1
                    cnt += 1
                    explore_queries.add(file_i) 

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)