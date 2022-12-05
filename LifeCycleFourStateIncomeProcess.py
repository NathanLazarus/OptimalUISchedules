import gurobipy as gp
import numpy as np
import scipy.sparse as sp
import sys
import pandas as pd

N_asset_states = 10
N_income_states = 4
N_consumption_choices = 10

Nstates = N_asset_states * N_income_states

job_losing_rate = 0.25
job_finding_rate = 0.25

income_state_transition_matrix = np.array(
    [
        [(1 - job_losing_rate), job_losing_rate, 0, 0],
        [job_finding_rate, 0, (1 - job_finding_rate), 0],
        [job_finding_rate, 0, 0, (1 - job_finding_rate)],
        [job_finding_rate, 0, 0, (1 - job_finding_rate)],
    ]
)

amin = 0
amax = 1
gamma = -10
beta = 0.9


high_income = 0.2
ui_1 = 0
ui_2 = 0
no_income = 0


def asset_income(k):
    return 0
    # return 2 * k - (k ** 2) / 2


def utility(consumption, gamma):
    return -np.exp(gamma * consumption)


def p_state_next_period(Nstates, all_states, asset_grid_space, remaining_assets, current_state, income_state_transition_matrix):
    probs = (
        ((asset_grid_space - abs(all_states.asset_values - remaining_assets)) / asset_grid_space) *
        income_state_transition_matrix[current_state[0], all_states.labor_income_index]
    )
    future_states = np.arange(Nstates)[probs > 1e-13]
    return future_states, probs[probs > 1e-13]


incomeDF = pd.DataFrame(
    {
        'labor_income_index': np.arange(N_income_states),
        'labor_income_values': np.array([high_income, ui_1, ui_2, no_income]),
        'dummy': 0
    }
)

assetDF = pd.DataFrame(
    {
        'asset_index': np.arange(N_asset_states),
        'asset_values': np.linspace(amin + amax / N_asset_states, amax, N_asset_states),
        'dummy': 0
    }
)

asset_grid_space = (amax - amin) / N_asset_states

all_states = incomeDF.merge(assetDF, how='cross')

all_states['asset_income_values'] = asset_income(all_states.asset_values)
all_states['total_money_at_beginning_of_period'] = all_states.labor_income_values + all_states.asset_values + all_states.asset_income_values
all_states['state_index'] = np.arange(all_states.shape[0])
# print(all_states.dtypes)

max_consumption = max(all_states.total_money_at_beginning_of_period) - min(all_states.asset_values)

consumptionDF = pd.DataFrame(
    {
        'consumption': np.linspace(0, max_consumption, N_consumption_choices),
        'dummy': 0
    }
)
consumptionDF['utility'] = utility(consumptionDF.consumption, gamma)

every_option = all_states.merge(consumptionDF, how='cross')

every_option['remaining_assets'] = every_option.total_money_at_beginning_of_period - every_option.consumption

valid_options = every_option[(every_option.remaining_assets >= (min(every_option.asset_values) - 1e-9)) & (every_option.remaining_assets <= (max(every_option.asset_values) + 1e-9))]
valid_options = valid_options.reset_index()

utilities = np.array(valid_options.utility)

Nconstraints = valid_options.shape[0]

test_dense_valueMat = np.zeros((Nconstraints, Nstates))

for row in valid_options.itertuples():
    test_dense_valueMat[row.Index, row.state_index] += 1
    future_states, p_transition_to_future_state = p_state_next_period(Nstates, all_states, asset_grid_space, row.remaining_assets, [row.labor_income_index, row.asset_index], income_state_transition_matrix)
    test_dense_valueMat[row.Index, future_states] += (-beta * p_transition_to_future_state)

valueMat = sp.csr_matrix(test_dense_valueMat)

try:

    # Create a new model
    m = gp.Model("UI")

    # Create variables
    x = m.addMVar(shape=Nstates, name="x", lb=-gp.GRB.INFINITY)

    # Set objective
    obj = np.ones(Nstates)
    m.setObjective(obj @ x, gp.GRB.MINIMIZE)

    # Add constraints
    m.addConstr(valueMat @ x >= utilities, name="constraint")

    # Use the dual simplex algorithm only (the default is to run
    # the barrier as well in parallel)
    m.setParam("Method", 1)
    m.setParam("Presolve", 1)

    # Optimize model
    m.optimize()

    print(x.X)
    print("Obj: %g" % m.objVal)

except gp.GurobiError as e:
    print("Error code " + str(e.errno) + ": " + str(e))

except AttributeError:
    print("Encountered an attribute error")


# Parameter Tuning:

# m.tune()

# if m.tuneResultCount > 0:

#     # Load the best tuned parameters into the model
#     m.getTuneResult(0)

#     # Write tuned parameters to a file
#     m.write('tune.prm')
