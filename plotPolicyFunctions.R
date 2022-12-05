policies = fread("policy_fun.csv")
setnames(policies, c("asset_values", "labor_income_values", "consumption", "reservation_wage", "foregone_utility"))

policies[, foregone_utility_of_best_choice := min(foregone_utility), .(asset_values, labor_income_values)]
actual_policies = policies[foregone_utility == foregone_utility_of_best_choice]

ggplot(actual_policies) +
  geom_line(aes(y = reservation_wage, x = asset_values, group = labor_income_values, color = labor_income_values), size = 0.9) +
  theme_cowplot() +
  # ylim(-0.05, 0) +
  ggtitle("Values") +
  theme(plot.title = element_text(hjust = 0),
        axis.title.y = element_blank()) #  +
  # colScale

ggsave("ReservationWagePolicyFunctions.png")

ggplot(actual_policies) +
  geom_line(aes(y = consumption, x = asset_values, group = labor_income_values, color = labor_income_values), size = 0.9) +
  theme_cowplot() +
  # ylim(-0.05, 0) +
  ggtitle("Values") +
  theme(plot.title = element_text(hjust = 0),
        axis.title.y = element_blank()) #  +
  # colScale


ggsave("ConsumptionPolicyFunctions.png")

