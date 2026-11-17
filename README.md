# District-Relative-Rent-Tier-Classification-in-Madrid
A Regression-to-Classification Pipeline with Tidymodels


Madrid’s rental market is strongly segmented by location: a flat in Salamanca can easily rent for twice
times a comparable property in Puente de Vallecas. This makes absolute price prediction less informative
for renters trying to assess whether a listing is good value. The question I want to answer is: given a
property’s observable features, is it priced Low, Fair, or High relative to its own district?
The approach is a two stage pipeline. First, I train regression models to predict rent_price directly. The
predicted value is then converted to a price tier using district specific quartile boundaries, these boundaries
are computed entirely on the training set, so there is no leakage from the test data. I compare three models:
Elastic Net, Random Forest, and XGBoost. All three are tuned using race-ANOVA elimination (Kuhn
2024) inside a unified workflow_set, which makes the comparison straightforward and reproducible.
