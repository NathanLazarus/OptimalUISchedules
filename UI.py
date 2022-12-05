import gurobipy as gp # Note: gurobipy requires Python 3.8 or *lower*
import numpy as np
import scipy.sparse as sp
import sys
import pandas as pd
import datetime


N_asset_states = 20
N_wage_states = 2
N_UI_states = 0
N_consumption_choices = 60

# Bounds on assets
amin = 0
amax = 20

r = 0.002  # monthly interest rate

gamma_exponential = -10
gamma_CRRA = 0.9
utility_function = "log"
beta = 0.95  # monthly discount rate

ui_benefits_level = 0.9
no_benefits = 0.001
wage_grid_space = np.sqrt(2)


asset_states = np.linspace(amin + amax / N_asset_states, amax, N_asset_states)
# wage_states = [wage_grid_space ** x for x in range(N_wage_states)] # normalize the lowest wage to 1
wage_states = [1, 10]





# The transitions between income states:
# In the employed state, with probability job_losing_rate, workers transition into the "unemployed for one month" state
# In the unemployed state, with probability job_finding_rate, the unemployed receive a job offer
# (this is independent of the duration of unemployment)

job_losing_rate = 0.1
job_finding_rate = 0.2
job_offer_dist = (np.array([job_finding_rate] * N_wage_states) / N_wage_states) # uniform probability over all wages

# The income states are N_wage_states employed (with income = wage_states) +
# N_UI_states of unemployment durations 0:(N_UI_states - 1) with income = ui_benefits_level
# + 1 of UI benefits expiration with income = no_benefits (some outside option)
N_income_states = N_wage_states + N_UI_states + 1
Nstates = N_asset_states * N_income_states

# labor_income_value_array = np.array(wage_states + [ui_benefits_level] * N_UI_states + [no_benefits])
labor_income_value_array = np.array(wage_states + [no_benefits])

assert len(labor_income_value_array) == N_income_states


def get_income_state_transition_matrix(job_offer_dist, reservation_wage):
    accepted_offers = np.zeros_like(job_offer_dist)
    accepted_offers[reservation_wage:] = job_offer_dist[reservation_wage:]
    job_taking_rate = np.sum(accepted_offers)

    transition_mat = np.zeros((N_wage_states + N_UI_states + 1, N_wage_states + N_UI_states + 1))
    transition_mat[:N_wage_states, :N_wage_states] = np.eye(N_wage_states) * (1 - job_losing_rate)
    transition_mat[:N_wage_states, N_wage_states] = job_losing_rate

    transition_mat[N_wage_states:, :N_wage_states] = accepted_offers
    transition_mat[N_wage_states : N_wage_states + N_UI_states, N_wage_states + 1 :] = np.eye(N_UI_states) * (1 - job_taking_rate)
    transition_mat[N_wage_states + N_UI_states, N_wage_states + N_UI_states] = (1 - job_taking_rate)

    return transition_mat





def asset_income(r, k):
    return r * k

if utility_function == "log":
    def utility(consumption, gamma_exponential, gamma_CRRA, utility_function):
        return np.log(consumption)
else:
    def utility(consumption, gamma_exponential, gamma_CRRA, utility_function):
        if utility_function == "exponential":
            u = -np.exp(gamma_exponential * consumption)
        elif utility_function == "CRRA":
            u = (1 / (1 - gamma_CRRA)) * consumption ** (1 - gamma_CRRA)
        else:
            sys.exit("utility function " + utility_function + " not defined")
        return u


def p_state_next_period(
    Nstates,
    all_states,
    asset_grid_space,
    remaining_assets,
    reservation_wage,
    current_state,
    job_offer_dist,
):
    probs = (
        ((asset_grid_space - abs(all_states.asset_values - remaining_assets)) / asset_grid_space) *
        get_income_state_transition_matrix(job_offer_dist, reservation_wage)[current_state[0], all_states.labor_income_index]
    )
    future_states = np.arange(Nstates)[probs > 1e-13]
    return future_states, probs[probs > 1e-13]



# create state DFs (the state space, all_states, is their Cartesian product)

incomeDF = pd.DataFrame(
    {
        "labor_income_index": np.arange(N_income_states),
        "labor_income_values": labor_income_value_array,
        "dummy": 0,
    }
)

assetDF = pd.DataFrame(
    {
        "asset_index": np.arange(N_asset_states),
        "asset_values": asset_states,
        "dummy": 0
    }
)

asset_grid_space = (amax - amin) / N_asset_states

all_states = incomeDF.merge(assetDF, how="cross")

all_states["asset_income_values"] = asset_income(r, all_states.asset_values)
all_states["total_money_at_beginning_of_period"] = (
    all_states.labor_income_values +
    all_states.asset_values +
    all_states.asset_income_values
)
all_states["state_index"] = np.arange(all_states.shape[0])

max_consumption = max(all_states.total_money_at_beginning_of_period) - min(all_states.asset_values)

# create control DFs (the control space is their Cartesian product)

consumptionDF = pd.DataFrame(
    {
        "consumption": np.linspace(0.001, max_consumption, N_consumption_choices),
        "dummy": 0
    }
)

reservation_wageDF = pd.DataFrame(
    {
        "reservation_wage": np.arange(N_wage_states + 1), # +1 because we allow for reservation wage > max wage
        "dummy": 0
    }
)

consumptionDF["utility"] = utility(consumptionDF.consumption, gamma_exponential, gamma_CRRA, utility_function)

# every_option is the Cartesian product of the controls with the states

every_option = all_states.merge(consumptionDF, how="cross").merge(reservation_wageDF, how="cross")

every_option["remaining_assets"] = every_option.total_money_at_beginning_of_period - every_option.consumption

# remove combinations of controls and states that take people off the asset grid
# avoid considering reservation wages for the employed because there's no on-the-job search

valid_options = every_option[
    (every_option.remaining_assets >= (min(every_option.asset_values) - 1e-9)) &
    (every_option.remaining_assets <= (max(every_option.asset_values) + 1e-9)) &
    ((every_option.labor_income_index >= N_wage_states) | (every_option.reservation_wage == 0))
]

valid_options = valid_options.reset_index()

utilities = np.array(valid_options.utility)

print(utilities.shape)

Nconstraints = valid_options.shape[0]

dense_valueMat = np.zeros((Nconstraints, Nstates))

# The Bellman equations V(x_i) >= u(c) + beta * E[x+] for all c
#                       V(x_i) >= u(c) + beta * sum(V(x+)p(x+|c))
#                       V(x_i) - beta * sum(V(x+)p(x+|c)) >= u(c)
# will enter as linear constraints, and hold with equality at the optimal c.
# Here we construct the matrix with entries 1 - beta * p(xi|c) at position i
# (p(xi|c) is the probability of staying in the current state)
# and -beta * p(x+|c) all other positions. When multiplied by the vector of values
# it will give the left hand side of the inequality for the linear programming problem

counter = 1
for row in valid_options.itertuples():
    dense_valueMat[row.Index, row.state_index] += 1
    future_states, p_transition_to_future_state = p_state_next_period(
        Nstates,
        all_states,
        asset_grid_space,
        row.remaining_assets,
        row.reservation_wage,
        [row.labor_income_index, row.asset_index],
        job_offer_dist,
    )
    counter = counter + 1

    dense_valueMat[row.Index, future_states] += (-beta * p_transition_to_future_state)

valueMat = sp.csr_matrix(dense_valueMat)

try:

    # Create the model
    m = gp.Model("UI")

    # Define variables
    values = m.addMVar(shape=Nstates, name="values", lb=-gp.GRB.INFINITY)

    # Objective: minimize the sum of the values (equivalent to minimizing each one on its own,
    # ensures that the Bellman constraint will hold with equality at the optimal control)
    obj = np.ones(Nstates)
    m.setObjective(obj @ values, gp.GRB.MINIMIZE)

    # Add Bellman constraints
    m.addConstr(valueMat @ values >= utilities, name="constraint")

    # Use dual simplex algorithm only (do not run barrier in parallel)
    m.setParam("Method", 1)
    m.setParam("Presolve", 1)

    # Run optimization
    m.optimize()

    # vector of values across income and asset states [(income_0, asset_0), (income_0, asset_1), ...]
    print("Sol:")
    print(values.X)

    # value of the objective: sum of values.X
    print("Obj: %g" % m.objVal)

    np.savetxt("values.csv", values.X, delimiter=",", comments="")

except gp.GurobiError as e:
    print("Error code " + str(e.errno) + ": " + str(e))

except AttributeError:
    print("Encountered an attribute error")


foregone_utility = valueMat @ values.X - utilities
policy_fun = np.hstack([valid_options[["asset_values", "labor_income_values", "consumption", "reservation_wage"]], foregone_utility[:,None]])[foregone_utility < 1e-8]
np.savetxt("policy_fun.csv", policy_fun, delimiter=",", comments="")