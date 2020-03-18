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
#----------read data----------#
sal_df = pd.read_csv("california-state-university-2018.csv")
sal_df = sal_df[sal_df["Total Pay & Benefits"] > 0]

# %% codecell
#----------test----------#
sns.distplot(sal_df['Total Pay & Benefits'], hist = False, kde = True, kde_kws = {'linewidth': 3})
plt.title("Total Pay (including benefits) of California University System Professors in 2018")
plt.xlabel("Total Pay and Benefits ($)")
plt.ylabel("Density")
plt.axvline(my_mean)

my_mean = sal_df["Total Pay & Benefits"].mean()
