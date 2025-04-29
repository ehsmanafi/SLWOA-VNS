import ast
import time
import numpy as np
import json
from Objective import CMax
from SLWOA_VNS_EXPECTED import SLWOA_VNS_EXPECTED

def load_from_json(filename):
    def convert_keys(obj):
        """ Recursively convert dictionary keys from string representations back to tuples where applicable. """
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                try:
                    # Convert back to tuple if possible
                    new_key = eval(k) if k.startswith("(") and k.endswith(")") else k
                except:
                    new_key = k
                new_dict[new_key] = convert_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [convert_keys(i) for i in obj]
        else:
            return obj

    with open(filename, 'r') as f:
        data = ast.literal_eval(json.load(f))
        return data[0],data[1],data[2],data[3],data[4],data[5],data[6]

Algo_name = 'SLWOA_VNS_EXPECTED'

print("Algo_name:  ",Algo_name)

iif = 'L08'
run_id = 10
tic = time.time()
Machine,Configuration,Job,R,RecChange,W,TRT = load_from_json("Problem_instances/Problem_Instance_{}.json".format(iif))
print('Start of ', "Problem_Instance_{}.json".format(iif))

Rp = {}

for i, job_count in Job.items():
    Rp[i] = {}
    for j in range(1, job_count + 1):
        keys, values = zip(*R[i, j].items())  # Extract keys and values

        values_np = np.array(values, dtype=np.float64)
        inv_values = 1.0 / values_np
        normalized = inv_values / inv_values.sum()
        cumulative = np.cumsum(normalized)

        # Zip keys with cumulative values and store
        Rp[i][j] = dict(zip(keys, cumulative))
for i, job_count in Job.items():
    for j in range(1, job_count + 1):
        sorted_items = sorted(Rp[i][j].items(), key=lambda item: item[1])  # Ensure deterministic order
        lfg = dict()
        current = np.float64(0.0)

        for key, prob in sorted_items:
            start = current
            end = prob
            lfg[key] = (start, end)
            current = end
        Rp[i][j] = lfg
job_start_index = {}
index = 0
for job_id, op_count in Job.items():
    job_start_index[job_id] = index
    index += op_count
lb=0.001
ub=0.999
dim=sum(list(Job.values()))*2+2
SearchAgents_no=100
Max_iteration=300
Best_score,Best_pos,WOA_cg_curve=SLWOA_VNS_EXPECTED(SearchAgents_no,Max_iteration,lb,ub,dim,CMax,Rp,job_start_index,Machine,Configuration,Job,R,RecChange,W,TRT,Algo_name,iif,run_id)
toc = time.time()
print(toc-tic)