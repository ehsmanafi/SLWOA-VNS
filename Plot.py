import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import openpyxl
from openpyxl.utils import get_column_letter
from Objective import select_by_probability_dict
from Objective import schedule_operations

def Plot(solution, R, Rp, job_start_index,Job,RecChange,Machine, W,TRT,Algo_name,iif,run_id):
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

    append_op = Machine_operations.__getitem__

    for i in range(total_jobs):
        job, op, (machine, config, worker) = Sequence[i]

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

    #return job_allocation, Machine_operations
    # -------- 🎨 Plotting --------
    fig, ax = plt.subplots(figsize=(14, 6))

    # Generate all unique titles with IDs
    all_titles = []
    for ops in Machine_operations.values():
        for op in ops:
            title = f"{op['Title']} (ID {op['ID']})"
            all_titles.append((op['ID'], title))

    # Sort titles by ID
    sorted_titles = sorted(all_titles, key=lambda x: x[0])
    title_colors = {title: (random.random(), random.random(), random.random()) for _, title in sorted_titles}

    # For legend
    legend_patches = []

    yticks = []
    ylabels = []

    for idx, (machine, ops) in enumerate(sorted(Machine_operations.items())):
        y = idx
        yticks.append(y)
        ylabels.append(f"Machine {machine}")
        for op in ops:
            start = op['start']
            end = op['end']
            width = end - start
            ID = op['ID']
            title_with_id = f"{op['Title']} (ID {ID})"
            color = title_colors[title_with_id]

            ax.barh(y, width, left=start, height=0.6, color=color, edgecolor='black')
            ax.text(start + width / 2, y, f"ID {ID}", va='center', ha='center', fontsize=8, color='white')

    # Sorted legend by ID
    for ID, title in sorted_titles:
        patch = mpatches.Patch(color=title_colors[title], label=title)
        legend_patches.append(patch)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Time")
    ax.set_title("Machine Schedule (Gantt Chart with IDs)")
    ax.grid(True, axis='x')
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig("{}_Machine_Schedule_Problem_Instance_{}_{}.jpeg".format(Algo_name,iif,run_id), dpi=300)
    plt.show()
    return Machine_operations