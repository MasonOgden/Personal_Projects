# %% codecell
#----------install packages----------#
!pip install -q pandas
!pip install -q numpy
!pip install -q matplotlib
!pip install -q seaborn

# %% codecell
#----------import packages----------#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %% codecell
# values in list start at July, end in March
util_data = {'water': [99.35, 42.55, 42.55, 72.17, 72.17, 72.17, 42.55, 72.17, 57.36],
             'electricity': [9.15, 11.37, 23.77, 9.93, 38.96, 32.15, 40.21, 40.22, 43.90],
             'gas': [36.77, 19.34, 18.61, 13.86, 17.44, 18.24, 14.07, 20.75, 17.80],
             'trash': [24.22, 24.22, 16.78, 16.78, 16.78, 16.78, 17.2, 17.2, 16.92],
             'internet': [45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0, 45.0],
             'month': ['July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March']}

df = pd.DataFrame(util_data, columns = ['month', 'water', 'electricity', 'gas', 'trash', 'internet'])

# %% codecell
df["total"] = df["water"] + df["electricity"] + df["gas"] + df["trash"] + df["internet"]
df_long = pd.melt(df, id_vars=['month'], value_vars=['water', 'electricity', 'gas', 'trash', 'internet', 'total'])
final_df = df_long.rename(columns={"variable": "Utility",
                                   "value": "Amount($)",
                                   "month": "Month"})
final_df["Month"] = final_df["Month"].astype("category")
final_df["Month"].cat.reorder_categories(['July', 'August', 'September', 'October', 'November', 'December', 'January', 'February', 'March'],
                                         inplace=True)

# %% codecell
sns.lineplot(x="Month", y="Amount($)", hue="Utility", data=final_df).set_title("Nine Months of Utility Payments")
sns.set(rc={'figure.figsize':(11, 6)})

# %% codecell
print(df[df["month"] != "July"]["total"].mean())
