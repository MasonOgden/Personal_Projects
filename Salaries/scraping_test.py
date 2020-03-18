# %% codecell
#----------install packages----------#
!pip install -q pandas
!pip install -q numpy
!pip install -q beautifulsoup4
!pip install -q requests
!pip install -q matplotlib
!pip install -q seaborn

# %% codecell
#----------import packages----------#
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import time
import matplotlib.pyplot as plt
import seaborn as sns

# %% codecell
#----------define variables----------#
# the link below is for 2018 data on professors only
link = 'https://transparentcalifornia.com/salaries/search/?q=Professor&y=2018&page='
pattern1 = re.compile(r'/([^/]*)/([^/]*)/.*')
pattern2 = re.compile(r'([^<]*)>([^<]*)')

# %% codecell
#----------populate data frame----------#
salary_df = pd.DataFrame(columns = ['system', 'name', 'title', 'total_pay_and_benefits'])

i=1
print("page: " + str(i))
page = requests.get(link + str(i))
soup = BeautifulSoup(page.text, 'html.parser')
table = soup.find('tbody')
done = (table is None)

while (not done):
    rows = table.find_all('tr')

    for row in rows:
        this_dict = {}
        this_dict["total_pay_and_benefits"] = float(row.find_all('td')[7].contents[0].replace('$', '').replace(',', ''))
        name_school_string = str(row.find_all('a')[0])[23:]
        title_string = str(row.find_all('a')[1])[25:]
        system_raw, name_raw = pattern1.match(name_school_string).groups()
        this_dict["system"] = system_raw.replace("-", " ")
        this_dict["name"] = name_raw.replace("-", " ")
        this_dict["title"] = pattern2.match(title_string).groups()[1]
        salary_df = salary_df.append(this_dict, ignore_index=True)

    i += 1
    print("page: " + str(i))
    page = requests.get(link + str(i))
    soup = BeautifulSoup(page.text, 'html.parser')
    table = soup.find('tbody')
    done = (table is None)
    #rows = table.find_all('tr')


    time.sleep(0.5)

# %% codecell
#----------output----------#
salary_df
salary_df[salary_df['system'] == 'california state university']
salary_df['system'].unique()

# %% codecell
#----------plot----------#
sns.distplot(salary_df['total_pay_and_benefits'], hist = False, kde = True, kde_kws = {'linewidth': 3})
plt.title("Total Pay (including benefits) of California University System Professors")
plt.xlabel("Total Pay and Benefits ($)")
plt.ylabel("Density")
