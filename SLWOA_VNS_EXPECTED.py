import numpy as np
import itertools
import time
from functools import partial
import random
from Plot import Plot
from VNS import variable_neighborhood_search_population

# Define state space discretization
AV10x = np.linspace(0, 1, 10)
AV25x = np.linspace(0, 1, 10)
d__x = np.linspace(0, 1, 10)

# Generate states using NumPy meshgrid for efficiency
states = np.array(np.meshgrid(range(len(AV10x) + 1),
                              range(len(AV25x) + 1),
                              range(len(d__x) + 1))).T.reshape(-1, 3)

# Initialize Q-table as a NumPy array instead of dictionary for faster access
Q = np.zeros((len(AV10x) + 1, len(AV25x) + 1, len(d__x) + 1, 8))

action_set = [(0.2, 0.5), (0.2, 1), (0.4, 0.5), (0.4, 1),
              (0.6, 0.5), (0.6, 1), (0.8, 0.5), (0.8, 1)]

def initialization(SearchAgents_no, dim):
    """Initialize random positions using NumPy for efficiency."""
    return np.random.random([SearchAgents_no,dim])
def getState(observation):
    """Digitize observation values efficiently using NumPy."""
    cartXdot = np.digitize(observation[0], AV10x)
    cartTheta = np.digitize(observation[1], AV25x)
    cartThed = np.digitize(observation[2], d__x)
    return (cartXdot, cartTheta, cartThed)
def maxAction(Q, state):
    """Find the action with the highest Q-value efficiently."""
    index = np.argmax(Q[state])
    return index,action_set[index]


def SLWOA_VNS_EXPECTED(SearchAgents_no,Max_iter,lb,ub,dim,fobj,Rp,job_start_index,Machine,Configuration,Job,R,RecChange,W,TRT,Algo_name,iif,run_id):
    np.random.seed(int(1000 * time.time()) % 2**32)
    # Initialize leader position and score
    Leader_pos = np.zeros(dim)
    Leader_score = float('inf')
    # Initialize search agents and ensure NumPy array for vectorized operations
    Positions=initialization(SearchAgents_no,dim)
    fobj_fixed = partial(fobj, Rp = Rp, job_start_index = job_start_index, Machine=Machine,Job=Job,R=R,RecChange=RecChange,W=W,TRT=TRT)
    fitness_values = np.apply_along_axis(fobj_fixed, 1, Positions)
    min_idx = np.argmin(fitness_values)
    Leader_score = fitness_values[min_idx]
    Leader_pos = Positions[min_idx].copy()
    d_base_initial = np.mean(fitness_values)
    d_initial = np.sum(fitness_values - d_base_initial)
    d__=1
    t=0
    Convergence_curve = np.zeros(Max_iter + 1)
    bestpop = [float('inf')]
    bestWOA = float('inf')
    reward = 0
    ALPHA = 0.5
    GAMMA = 0.6
    EPS = 1.0
    observation = np.array([0, 0, 1], dtype=np.float32)
    s = getState(observation)
    rand = np.random.random()
    mean=0
    start_time = time.time()
    while time.time() - start_time < Max_iter:
        Best_score = Leader_score
        mean1=mean
        Positions = np.mod(Positions, 1) 
        fitnesses = np.apply_along_axis(fobj_fixed, 1, Positions)
        mean = np.mean(fitnesses)
        bestWOA = Leader_score
        bestpop.append(Leader_score)
        recent_10 = np.mean(bestpop[-10:])
        recent_25 = np.mean(bestpop[-25:])
        AV10 = abs((bestWOA - recent_10) / recent_10)
        AV25 = abs((bestWOA - recent_25) / recent_25)
        fitnesses = np.apply_along_axis(fobj_fixed, 1, Positions)
        d_base = np.mean(fitnesses)
        d_ = np.sum(fitnesses - d_base)
        d__ = abs(d_ / d_initial)
        # Construct observation as a float32 NumPy array
        observation_ = np.array((AV10, AV25, d__), dtype=np.float32)
        # Decision logic
        s_ = getState(observation_)
        # ε-greedy action selection
        rand = np.random.random()
        (index, aa) = maxAction(Q, s_) if rand < (1 - EPS) else (lambda i: (i, action_set[i]))(np.random.choice(8))
        # Update pp and coefficients `a` and `a2`
        pp = aa[0]
        a = 2 - t * (2 / Max_iter)
        a2 = -1 + t * (-1 / Max_iter)
        n_agents, dim = Positions.shape
        r1 = np.random.rand(n_agents)
        r2 = np.random.rand(n_agents)
        A = 2 * a * r1[:, np.newaxis]
        C = 2 * r2[:, np.newaxis]
        b = 1
        l = (a2 - 1) * np.random.rand(n_agents) + 1
        p = np.random.rand(n_agents)

        for i in range(n_agents):
            if p[i] < pp:
                if abs(A[i, 0]) >= aa[1]:
                    rand_index = int(np.floor(SearchAgents_no * np.random.rand()))
                    X_rand = Positions[rand_index]
                    D_X_rand = np.abs(C[i] * X_rand - Positions[i])
                    Positions[i] = X_rand - A[i] * D_X_rand
                else:
                    D_Leader = np.abs(C[i] * Leader_pos - Positions[i])
                    Positions[i] = Leader_pos - A[i] * D_Leader
            else:
                distance2Leader = np.abs(Leader_pos - Positions[i])
                Positions[i] = (distance2Leader *
                                np.exp(b * l[i]) *
                                np.cos(2 * np.pi * l[i]) +
                                Leader_pos)
        Positions = np.mod(Positions, 1) 
        # Apply VNS to all agents at once
        Positions = variable_neighborhood_search_population(Positions, fobj_fixed, k_max=3, step_size=0.4)
        # Recompute fitness and update Leader
        fitness_values = np.array([fobj_fixed(pos) for pos in Positions])
        mean = np.mean(fitnesses)
        min_idx = np.argmin(fitness_values)
        min_val = fitness_values[min_idx]

        if min_val < Leader_score:
            Leader_score = min_val
            Leader_pos = Positions[min_idx].copy()
        t=t+1
        # Reinforcement learning reward logic
        reward = 0
        if Leader_score < Best_score:
            reward += 10
        if mean < mean1:
            reward += 2
        # ==== Expected SARSA Update ====
        q_values = Q[s_]
        q_max = np.max(q_values)
        greedy_mask = (q_values == q_max)
        num_greedy = np.sum(greedy_mask)

        # Avoid division by zero
        if num_greedy == 0:
            greedy_mask = np.ones_like(q_values, dtype=bool)
            num_greedy = len(q_values)
        non_greedy_prob = EPS / Q.shape[1]
        greedy_prob = ((1 - EPS) / num_greedy) + non_greedy_prob

        probs = np.where(greedy_mask, greedy_prob, non_greedy_prob)
        expected_q = np.sum(probs * q_values)

        Q[s_, index] += ALPHA * (reward + GAMMA * expected_q - Q[s_, index])
        # Decay exploration rate
        EPS = EPS - 1 / Max_iter if EPS > 0 else 0
        # Logging
        print(f'Iteration: {t}    Leader_score: {Leader_score}')
    Plot(Leader_pos, R, Rp, job_start_index,Job,RecChange,Machine, W, TRT,Algo_name,iif,run_id)
    return Leader_score,Leader_pos,Convergence_curve