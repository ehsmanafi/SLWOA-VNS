import numpy as np
import itertools
import time
from functools import partial
import random

def schedule_operations(jobs, priority_array,job_start_index):
    # Step 2: Initialize current operation index for each job
    current_op_idx = {job_id: 0 for job_id in jobs}
    total_ops = sum(jobs.values())
    schedule = []

    # Step 3: Loop to build the schedule
    for _ in range(total_ops):
        min_val = float('inf')
        selected_job = None

        for job_id in jobs:
            if current_op_idx[job_id] < jobs[job_id]:  # has remaining ops
                op_position = job_start_index[job_id] + current_op_idx[job_id]
                value = priority_array[op_position]
                if value < min_val:
                    min_val = value
                    selected_job = job_id

        # Append the selected operation to the schedule
        schedule.append((selected_job, current_op_idx[selected_job]))
        current_op_idx[selected_job] += 1

    return schedule

def select_by_probability_dict(prob_dict, value):
    for key, (start, end) in prob_dict.items():
          if start <= value < end:  # end is exclusive
              return key
    return None  # Value not found in any interval

def CMax(solution,Rp,job_start_index,Machine,Job,R,RecChange,W,TRT,cmx1 = False, cmx2 = False, cmx3 = False):
    total_jobs = sum(Job.values())
    Scheduling = solution[:total_jobs]
    MachineConfiguration = solution[total_jobs:2 * total_jobs]

    schedule = schedule_operations(Job, Scheduling, job_start_index)
    Sequence = [None] * total_jobs

    for idx in range(total_jobs):
        job = schedule[idx][0]
        op = schedule[idx][1] + 1
        conf_idx = job_start_index[job] + op - 1
        conf = select_by_probability_dict(Rp[job][op], MachineConfiguration[conf_idx])
        Sequence[idx] = (job, op, conf)

    # Precompute dictionaries
    SCT = RecChange  # Already structured as {machine: {(c1, c2): time}}
    PT_lookup = {
        (j, o, w, m, c): R[(j, o)][(m, c, w)]
        for (j, o) in R
        for (m, c, w) in R[(j, o)]
    }

    # Setup
    machine_avail = {m: 0 for m in Machine}
    worker_avail = {w: 0 for w in W}
    job_avail = {j: 0 for j in Job}
    machine_config = {m: None for m in Machine}
    worker_machine = {w: None for w in W}
    job_machine = {j: None for j in Job}
    job_allocation = [None] * total_jobs
    Machine_operations = {m: [] for m in Machine}
    operations_on_machine = {m: [] for m in Machine}
    job_bottleneck = {j: 0 for j in Job}

    # Helper functions
    append_op = Machine_operations.__getitem__

    for i in range(total_jobs):
        job, op, (machine, config, worker) = Sequence[i]
        operations_on_machine[machine].append(job_start_index[job]+op-1)

        m_avail = machine_avail[machine]
        w_avail = worker_avail[worker]
        j_avail = job_avail[job]

        # Config change time
        prev_conf = machine_config[machine]
        if prev_conf is not None and prev_conf != config:
            m_avail += SCT[machine][(prev_conf, config)]
            machine_config[machine] = config
        elif prev_conf is None:
            machine_config[machine] = config

        # Worker travel
        prev_machine_w = worker_machine[worker]
        if prev_machine_w is not None and prev_machine_w != machine:
            w_avail += TRT[(prev_machine_w, machine)]
            worker_machine[worker] = machine
        elif prev_machine_w is None:
            worker_machine[worker] = machine

        # Job transfer
        prev_machine_j = job_machine[job]
        if prev_machine_j is not None and prev_machine_j != machine:
            j_avail += TRT[(prev_machine_j, machine)]
            job_machine[job] = machine
        elif prev_machine_j is None:
            job_machine[job] = machine

        # Compute operation times
        start_time = max(m_avail, w_avail, j_avail)
        job_bottleneck[job] = max(job_bottleneck[job], start_time-m_avail)
        proc_time = PT_lookup[(job, op, worker, machine, config)]
        end_time = start_time + proc_time

        # Update availability
        machine_avail[machine] = end_time
        worker_avail[worker] = end_time
        job_avail[job] = end_time

        # Store allocation
        job_allocation[i] = (job, op, machine, config, worker, start_time, end_time)

        # Add to machine operation log
        append_op(machine).append({
            'start': start_time,
            'end': end_time,
            'Title': f'J:{job}, Op:{op}, W:{worker}, M:{machine}, C:{config}, S:{start_time}, E:{end_time}',
            'ID': i
        })
    if cmx1 ==True:
        dim = solution.shape[0]
        all_ops = []
        for machine, ops in Machine_operations.items():
            for op in ops:
                # Parse job and op from title if not already separated
                if 'J' not in op or 'O' not in op:
                    # Extract from title string
                    parts = {kv.split(':')[0]: kv.split(':')[1] for kv in op['Title'].split(', ')}
                    job = int(parts['J'])
                    operation = int(parts['Op'])
                else:
                    job = op['J']
                    operation = op['O']
                all_ops.append({
                    'machine': machine,
                    'job': job,
                    'op': operation,
                    'start': op['start'],
                    'end': op['end'],
                })

        # Now find overlaps between every pair
        overlapping_pairs = []

        for i in range(len(all_ops)):
            for j in range(i + 1, len(all_ops)):
                op1 = all_ops[i]
                op2 = all_ops[j]

                # Skip if same job-op or on same machine (optional)
                if (op1['job'], op1['op']) == (op2['job'], op2['op']):
                    continue

                # Check for time overlap
                if op1['start'] < op2['end'] and op2['start'] < op1['end']:
                    overlapping_pairs.append((
                        (op1['job'], op1['op'], op1['machine']),
                        (op2['job'], op2['op'], op2['machine'])
                    ))

        # Print overlaps
        overlapps = list()
        for pair in overlapping_pairs:
            overlapps.append((dim // 2 + job_start_index[pair[0][0]] + pair[0][1] - 1 - 1,dim // 2 + job_start_index[pair[1][0]] + pair[1][1] - 1 - 1))
        return overlapps
    #obje = max(machine_avail.values())

    key_with_max_value, obje = max(machine_avail.items(), key=lambda item: item[1])
    if cmx2 ==True:
        return operations_on_machine[key_with_max_value]
    if cmx3 ==True:
        ggg, xxx = max(job_bottleneck.items(), key=lambda item: item[1])
        return [job_start_index[ggg]+igh for igh in range(Job[ggg])]
    return obje