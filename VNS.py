import numpy as np
import itertools
import time
from functools import partial
import random
from Plot import Plot


def HNS(Positions, fobj, step_size=0.01, attempts=1):
    """
    Parameters:
        Positions: ndarray (n_agents, dim), values in [0, 1]
        fobj: objective function, takes 1D array and returns scalar
        step_size: max noise per perturbed dimension
        attempts: number of local perturbations per agent

    Returns:
        Improved Positions (ndarray of same shape)
    """
    n_agents, dim = Positions.shape
    best_positions = Positions.copy()
    best_scores = np.array([fobj(pos) for pos in best_positions])

    for _ in range(attempts):
        # Generate random perturbation mask
        perturb_mask = np.random.rand(n_agents, dim) < 0.2  # ~20% dims changed
        noise = np.random.uniform(-step_size, step_size, size=(n_agents, dim))
        perturbation = perturb_mask * noise

        candidates = np.mod(best_positions + perturbation, 1.0)

        # Evaluate all candidates
        candidate_scores = np.array([fobj(candidate) for candidate in candidates])

        # Accept better ones
        improved_mask = candidate_scores < best_scores
        best_positions[improved_mask] = candidates[improved_mask]
        best_scores[improved_mask] = candidate_scores[improved_mask]
    Positions = best_positions.copy()
    xllms = Positions.copy()

    for _ in range(attempts):
        n_agents, n_dim = xllms.shape

        # 1. Collect overlaps for all agents
        overlaps_list = [fobj(xllm, cmx1=True) for xllm in xllms]

        # 2. Sample one (ii, jj) pair per agent
        selected_pairs = np.array([
            overlaps[np.random.choice(len(overlaps))] if overlaps else (-1, -1)
            for overlaps in overlaps_list
        ])
        valid_mask = selected_pairs[:, 0] != -1

        # 3. Generate 5 perturbations for each agent (shape: [n_agents, 5, n_dim])
        x_batch = np.repeat(xllms[:, np.newaxis, :], 5, axis=1)

        # 4. Generate random perturbations between 0.1 and 0.3
        perturb_ii = np.random.uniform(0.1, 0.3, size=(n_agents, 5))
        perturb_jj = np.random.uniform(0.1, 0.3, size=(n_agents, 5))

        # 5. Apply perturbations using advanced indexing
        for i in range(n_agents):
            ii, jj = selected_pairs[i]
            if ii == -1: continue  # skip invalid
            x_batch[i, :, ii] += perturb_ii[i]
            x_batch[i, :, jj] += perturb_jj[i]

        # 6. Modulo all perturbed candidates
        x_batch = np.mod(x_batch, 1.0)

        # 7. Flatten for batch evaluation
        flat_candidates = x_batch.reshape(-1, n_dim)
        flat_scores = np.array([fobj(c) for c in flat_candidates])
        scores = flat_scores.reshape(n_agents, 5)

        # 8. Select best perturbation per agent
        best_idxs = np.argmin(scores, axis=1)
        best_perturbations = x_batch[np.arange(n_agents), best_idxs]

        # 9. Keep best if improved
        current_scores = np.array([fobj(x) for x in xllms])
        improved = (scores[np.arange(n_agents), best_idxs] <= current_scores) & valid_mask
        xllms[improved] = best_perturbations[improved]

        # 10. Final evaluation for updated agents
        candidates = np.mod(xllms, 1.0)
        candidate_scores = np.array([fobj(c) for c in candidates])

        # 11. Update global bests
        improved_mask = candidate_scores < best_scores
        best_positions[improved_mask] = candidates[improved_mask]
        best_scores[improved_mask] = candidate_scores[improved_mask]
    for _ in range(attempts):
        new_Positions = Positions.copy()

        # 1. Vectorized: gather operations_on_machine for all positions
        operations_on_machine_list = [fobj(pos, cmx2=True) for pos in Positions]

        # 2. Flatten all (row-wise) updates
        lengths = np.array([len(ops) for ops in operations_on_machine_list])
        if lengths.sum() > 0:
            row_idx = np.repeat(np.arange(len(Positions)), lengths)
            col_idx = np.concatenate([
                np.array(ops, dtype=int) + dim // 2
                for ops in operations_on_machine_list if ops
            ])
            perturb_vals = np.random.rand(len(col_idx))

            # 3. Apply all updates in one shot
            np.add.at(new_Positions, (row_idx, col_idx), perturb_vals)

        # 4. Evaluate and update
        candidates = np.mod(new_Positions, 1.0)
        candidate_scores = np.array([fobj(c) for c in candidates])

        improved_mask = candidate_scores < best_scores
        best_positions[improved_mask] = candidates[improved_mask]
        best_scores[improved_mask] = candidate_scores[improved_mask]
    for _ in range(attempts):
        new_Positions = Positions.copy()

        # 1. Vectorized: gather operations_on_machine for all positions
        CMX_indecies = [fobj(pos, cmx3=True) for pos in Positions]
        # 2. Flatten all (row-wise) updates
        lengths = np.array([len(ops) for ops in CMX_indecies])
        if lengths.sum() > 0:
            row_idx = np.repeat(np.arange(len(Positions)), lengths)
            col_idx = np.concatenate([
                np.array(ops, dtype=int)
                for ops in CMX_indecies if ops
            ])
            perturb_vals = np.random.rand(len(col_idx))

            # 3. Apply all updates in one shot
            np.add.at(new_Positions, (row_idx, col_idx), perturb_vals)

        # 4. Evaluate and update
        candidates = np.mod(new_Positions, 1.0)
        candidate_scores = np.array([fobj(c) for c in candidates])

        improved_mask = candidate_scores < best_scores
        best_positions[improved_mask] = candidates[improved_mask]
        best_scores[improved_mask] = candidate_scores[improved_mask]
    for _ in range(attempts):
        new_Positions = Positions.copy()

        # 1. Vectorized: gather operations_on_machine for all positions
        CMX_indecies = [fobj(pos, cmx3=True) for pos in Positions]
        # 2. Flatten all (row-wise) updates
        lengths = np.array([len(ops) for ops in CMX_indecies])
        if lengths.sum() > 0:
            row_idx = np.repeat(np.arange(len(Positions)), lengths)
            col_idx = np.concatenate([
                np.array(ops, dtype=int) + dim // 2
                for ops in CMX_indecies if ops
            ])
            perturb_vals = np.random.rand(len(col_idx))

            # 3. Apply all updates in one shot
            np.add.at(new_Positions, (row_idx, col_idx), perturb_vals)

        # 4. Evaluate and update
        candidates = np.mod(new_Positions, 1.0)
        candidate_scores = np.array([fobj(c) for c in candidates])

        improved_mask = candidate_scores < best_scores
        best_positions[improved_mask] = candidates[improved_mask]
        best_scores[improved_mask] = candidate_scores[improved_mask]
    return best_positions
def variable_neighborhood_search_population(Positions, fobj, k_max=3, step_size=0.01):
    """
    Vectorized Variable Neighborhood Search (VNS) for a population.
    Applies VNS to only the top 20% of the agents based on their fitness.

    Parameters:
        Positions: ndarray (n_agents, dim), initial solutions in [0, 1]
        fobj: objective function taking a 1D array and returning scalar
        k_max: max neighborhood size
        step_size: local search perturbation strength

    Returns:
        Updated Positions after VNS
    """
    n_agents, dim = Positions.shape
    best_positions = Positions.copy()
    best_scores = np.array([fobj(pos) for pos in best_positions])

    # Sort agents based on their fitness scores
    sorted_indices = np.argsort(best_scores)

    # Get the top 20% of agents
    top_20_percent_idx = sorted_indices[:int(0.2 * n_agents)]
    bottom_80_percent_idx = sorted_indices[int(0.2 * n_agents):]

    # Initialize k_values for each agent
    k_values = np.ones(n_agents, dtype=int)  # Track neighborhood size per agent

    # Apply VNS to the top 20% agents
    top_20_positions = best_positions[top_20_percent_idx]
    top_20_scores = best_scores[top_20_percent_idx]

    while np.any(k_values[top_20_percent_idx] <= k_max):
        shaken = top_20_positions.copy()

        # Apply local search to the shaken solutions (VNS)
        local_searched = HNS(shaken, fobj, step_size=step_size, attempts=1)

        # Evaluate fitness of the local searched solutions
        candidate_scores = np.array([fobj(candidate) for candidate in local_searched])

        # Update best positions for the top 20%
        improved_mask = candidate_scores < top_20_scores
        top_20_positions[improved_mask] = local_searched[improved_mask]
        top_20_scores[improved_mask] = candidate_scores[improved_mask]

        # Increase k for solutions that did not improve
        k_values[top_20_percent_idx] = np.where(improved_mask, 1, k_values[top_20_percent_idx] + 1)

    # Merge the updated top 20% and unchanged bottom 80%
    best_positions[top_20_percent_idx] = top_20_positions
    return best_positions
