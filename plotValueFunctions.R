library(data.table)
library(ggplot2)
library(viridis)
library(cowplot)


cartesian_prod = function(X,Y) {
  stopifnot(is.data.table(X),is.data.table(Y))
  k = NULL
  X = X[, c(k = 1, .SD)]
  setkey(X, k)
  Y = Y[, c(k = 1, .SD)]
  setkey(Y, NULL)
  X[Y, allow.cartesian = TRUE][, k := NULL][]
}


value_sols = fread("values.csv")



N_wage_states = 2
N_UI_states = 0
N_asset_states = 30 # nrow(value_sols) / (N_wage_states + N_UI_states)
# N_consumption_choices = 90

# Bounds on assets
amin = 0
amax = 5

r = 0.002  # monthly interest rate

gamma_exponential = -10
gamma_CRRA = 0.9
utility_function = "exponential"
beta = 0.95  # monthly discount rate

ui_benefits_level = 0.9
no_benefits = 0.3
wage_grid_space = sqrt(2)

wages = c(1, 10)

income_value = c(wages, rep(ui_benefits_level, N_UI_states), no_benefits)
income_str =
  c(
    rep("Wage: ", N_wage_states),
    paste0(rep("Month: ", N_UI_states), seq_len(N_UI_states), rep("\nBenefit: ", N_UI_states)),
    "No Benefits: "
  )

wages = data.table(earnings = income_value, wage_state = paste0(income_str, income_value))

# wages = data.table(index = 1:(N_wage_states + N_UI_states + 1)
#                  )[1:N_wage_states,
#                    `:=`(earnings = wage_grid_space ^ (index - 1),
#                         wage_state = paste0("Wage: ", round(wage_grid_space ^ (index - 1), 2)))
#                  ][(N_wage_states + 1):(N_wage_states + N_UI_states),
#                    `:=`(earnings = ui_benefits_level,
#                         wage_state = paste0("Benefit: ", ui_benefits_level, "\nMonth: ", seq_len(N_UI_states)))
#                  ][N_wage_states + N_UI_states + 1,
#                    `:=`(earnings = no_benefits,
#                         wage_state = paste0("No Benefits (", no_benefits, ")"))]
#

assets = data.table(assets = amin + amax * (1:N_asset_states) / N_asset_states)


value_data = cartesian_prod(assets, wages)[, values := value_sols$V1]




myColors = viridis(n = (N_wage_states + N_UI_states + 1) * 2, option = "turbo")[c(N_wage_states:1, (N_wage_states * 2 + 1):(N_wage_states * 2 + N_UI_states), (N_wage_states + N_UI_states + 1) * 2)]
names(myColors) = wages$wage_state
colScale = scale_colour_manual(name = "wage_state", values = myColors)


ggplot(value_data) +
  geom_line(aes(y = values, x = assets, group = wage_state, color = wage_state), size = 0.9) +
  theme_cowplot() +
  # ylim(-0.05, 0) +
  ggtitle("Values") +
  theme(plot.title = element_text(hjust = 0),
        axis.title.y = element_blank()) #  +
  # colScale

# ggsave("ValueFunctions.png")

