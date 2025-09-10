import pandas as pd

df = pd.read_csv("Results/results.csv")
df.style.set_table_styles(
    [{'selector': 'th', 'props': [('background-color', '#f7f7f7')]}]
)

df = pd.DataFrame({'Model': ['Homogenous Poisson (Baseline)', 'Poisson GLM', 'Hawkes StdDiff w/ GLM Base', 'Poisson MLP (2 hidden layers 20 wide)'], 'Log Lik': [-22857.804/5, -20887.781/5, -20843.948/5, -19555.218/5]})
df.style.set_table_styles(
    [{'selector': 'th', 'props': [('background-color', '#f7f7f7')]}]
)

df = pd.DataFrame({'Model': ['Homogenous Poisson (Baseline)', 'Poisson GLM', 'Hawkes StdDiff w/ GLM Base'], 'Log Lik': [-8786.6729/2, -7954.551/2, -7922.531/2]})
df.style.set_table_styles(
    [{'selector': 'th', 'props': [('background-color', '#f7f7f7')]}]
)
