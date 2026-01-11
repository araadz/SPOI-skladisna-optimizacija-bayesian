import numpy as np
import pandas as pd
import pulp

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


# ============================================================
# DEFAULT PARAMETRI
# ============================================================

DEFAULT_PARAMS = {
    'COST_A': 2.0,
    'COST_B': 5.0,
    'COST_C': 2.0,
    'TAU': 8.0,
    'DEMAND_MULTIPLIER': 15,
    'N_PICKS': 1000
}


# ============================================================
# COST / QUALITY / UTILITY
# ============================================================

def calculate_cost(H, V, E, params):
    return (
        params['COST_A'] * H +
        params['COST_B'] * max(0, V - 2) +
        params['COST_C'] * (4 - E)
    )


def calculate_quality(H, V, E, params):
    return np.exp(-calculate_cost(H, V, E, params) / params['TAU'])


def calculate_utility(izlaz_norm, H, V, E, params):
    return calculate_quality(H, V, E, params) * (
        1 + izlaz_norm * params['DEMAND_MULTIPLIER']
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    df = df.dropna(subset=['H', 'V', 'E', 'izlaz']).reset_index(drop=True)
    if 'TEZINA_KAT' not in df.columns:
        df['TEZINA_KAT'] = 4
    df['TEZINA_KAT'] = df['TEZINA_KAT'].fillna(4)

    max_izlaz = df['izlaz'].max() if df['izlaz'].max() > 0 else 1
    df['izlaz_norm'] = df['izlaz'] / max_izlaz
    return df


# ============================================================
# UTILITY MATRIX
# ============================================================

def generate_utility_matrix(df, df_pos, params):
    n = len(df)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            U[i, j] = calculate_utility(
                df.iloc[i]['izlaz_norm'],
                df_pos.iloc[j]['H'],
                df_pos.iloc[j]['V'],
                df_pos.iloc[j]['E'],
                params
            )
    return U


# ============================================================
# ILP SOLVER
# ============================================================

def solve_ilp(U, df, df_pos):
    n = len(df)
    prob = pulp.LpProblem("Warehouse", pulp.LpMaximize)

    x = pulp.LpVariable.dicts(
        "x", ((i, j) for i in range(n) for j in range(n)), cat="Binary"
    )

    prob += pulp.lpSum(U[i, j] * x[i, j] for i in range(n) for j in range(n))

    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n)) == 1

    for j in range(n):
        prob += pulp.lpSum(x[i, j] for i in range(n)) <= 1

    heavy = df[df['TEZINA_KAT'] >= 4].index.tolist()
    high_v = df_pos[df_pos['V'] > 3].index.tolist()

    for i in heavy:
        for j in high_v:
            prob += x[i, j] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return {i: j for i in range(n) for j in range(n) if pulp.value(x[i, j]) == 1}


# ============================================================
# SIMULATION
# ============================================================

def simulate_picks(assign, df, df_pos, params):
    n = len(df)

    utils = np.array([
        calculate_utility(
            df.iloc[i]['izlaz_norm'],
            df_pos.iloc[assign[i]]['H'],
            df_pos.iloc[assign[i]]['V'],
            df_pos.iloc[assign[i]]['E'],
            params
        ) for i in range(n)
    ])

    costs = np.array([
        calculate_cost(
            df_pos.iloc[assign[i]]['H'],
            df_pos.iloc[assign[i]]['V'],
            df_pos.iloc[assign[i]]['E'],
            params
        ) for i in range(n)
    ])

    izlaz = df['izlaz'].values
    probs = izlaz / izlaz.sum()

    np.random.seed(42)
    picked = np.random.choice(n, size=params['N_PICKS'], p=probs)
    sim_cost = costs[picked].sum()

    wH = np.sum(df_pos.iloc[list(assign.values())]['H'].values * izlaz) / izlaz.sum()
    wV = np.sum(df_pos.iloc[list(assign.values())]['V'].values * izlaz) / izlaz.sum()

    return utils, costs, sim_cost, wH, wV


# ============================================================
# CORE OPTIMIZATION (ILP)
# ============================================================

def optimize(df, params):
    df = prepare_data(df)
    df_pos = df[['H', 'V', 'E']].reset_index(drop=True)
    n = len(df)

    init_assign = {i: i for i in range(n)}
    init_utils, init_costs, init_sim, init_wH, init_wV = simulate_picks(
        init_assign, df, df_pos, params
    )

    U = generate_utility_matrix(df, df_pos, params)
    opt_assign = solve_ilp(U, df, df_pos)

    opt_utils, opt_costs, opt_sim, opt_wH, opt_wV = simulate_picks(
        opt_assign, df, df_pos, params
    )

    return {
        'df': df,
        'df_positions': df_pos,
        'n_items': n,
        'params': params,

        'init_assign': init_assign,
        'init_utils': init_utils,
        'init_costs': init_costs,
        'init_sim': init_sim,
        'init_wH': init_wH,
        'init_wV': init_wV,

        'opt_assign': opt_assign,
        'opt_utils': opt_utils,
        'opt_costs': opt_costs,
        'opt_sim': opt_sim,
        'opt_wH': opt_wH,
        'opt_wV': opt_wV,

        'improvement': (opt_utils.sum() - init_utils.sum()) / init_utils.sum() * 100,
        'cost_reduction': (init_sim - opt_sim) / init_sim * 100,
        'h_reduction': (init_wH - opt_wH) / init_wH * 100,
        'v_reduction': (init_wV - opt_wV) / init_wV * 100,
        'moved': sum(1 for i in range(n) if opt_assign[i] != i)
    }


# ============================================================
# BAYESIAN OPTIMIZATION OF PARAMETERS
# ============================================================

SPACE = [
    Real(0.5, 5.0, name='COST_A'),
    Real(1.0, 10.0, name='COST_B'),
    Real(0.5, 5.0, name='COST_C'),
    Real(2.0, 20.0, name='TAU'),
    Integer(5, 30, name='DEMAND_MULTIPLIER')
]


def bayesian_optimize_parameters(df, n_trials=10, progress_callback=None):
    df_p = prepare_data(df)
    df_pos = df_p[['H', 'V', 'E']].reset_index(drop=True)

    @use_named_args(SPACE)
    def objective(**x):
        params = DEFAULT_PARAMS.copy()
        params.update(x)
        U = generate_utility_matrix(df_p, df_pos, params)
        return -U.sum()

    def callback(res):
        if progress_callback:
            progress_callback(len(res.x_iters), n_trials, -res.fun)

    res = gp_minimize(
        objective,
        SPACE,
        n_calls=n_trials,
        random_state=42,
        callback=[callback]
    )

    best_params = DEFAULT_PARAMS.copy()
    for k, v in zip([d.name for d in SPACE], res.x):
        best_params[k] = v

    return best_params, -res.fun


# ============================================================
# EXPORT
# ============================================================

def create_output_dataframe(results):
    df = results['df'].copy()
    pos = results['df_positions']
    opt = results['opt_assign']
    n = results['n_items']

    df['NOVI_H'] = [int(pos.iloc[opt[i]]['H']) for i in range(n)]
    df['NOVI_V'] = [int(pos.iloc[opt[i]]['V']) for i in range(n)]
    df['NOVI_E'] = [int(pos.iloc[opt[i]]['E']) for i in range(n)]
    df['utility'] = results['opt_utils']
    df['position_cost'] = results['opt_costs']

    if 'izlaz_norm' in df.columns:
        df = df.drop(columns=['izlaz_norm'])

    return df
