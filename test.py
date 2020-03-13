# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statistics 
import os

filepath = os.path.join(os.getcwd(),'CSV_Files','ninja_reports.xls')
df = pd.read_excel('ninja_reports2.xls',sheet_name='ninja_reports')

# %%
df.head(40)

# %%
city_ids = list(df.city_id.unique())
print(city_ids)

# %%
df.count()

# %%
df = df.drop(columns = ['jg_sub_category','title_id'])

# %%
#Correlation Matrix
corr = df.corr()
plt.figure(num=None, figsize=(7, 6), dpi=80, facecolor='w', edgecolor='k')
corrMat = plt.matshow(corr, fignum = 1)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.gca().xaxis.tick_bottom()
plt.colorbar(corrMat)
plt.title(f'Correlation Matrix', fontsize=15)
plt.show()


# %%
def isint(n):
    try:
        int(n)
        return n
    except:
        return -1;

ward_count = 0
for i in df.ward_id:
    if(i<=198):
        ward_count+=1
print("Noise: ",ward_count)

# %%


# %%
l = list(df.ward_id.unique())
len(l)

# %%
c_count = 0
for i in df.location:
    if pd.isna(i):
            c_count+=1
print("Count : ",c_count)

# %%
title = list(df.title.unique())
print(title,len(title))

# %%
cat = {}
for i in df.category.unique():
    cat[i] = 0

for i in df.category:
    cat[i] +=1

print(cat)
plt.figure(figsize = (7,7))
plt.title("Categories")
plt.pie(cat.values(),labels = cat.keys(),autopct = '%.2f')

# %%



# %%
chennai = [str(i) for i in df.location if 'Chennai' in str(i)]
for i in chennai:
    df.drop(d)

# %%


# %%
for i in df.category:
    cat[i] +=1

print(cat)

# %%
df.createdAt = pd.to_datetime(df['createdAt'])
plt.figure(figsize=(13,7))
plt.title("Complaint Dates: During what time were most complaints made?")
plt.xlabel('Dates')
plt.grid(True)
plt.ylabel('Count')
plt.hist(df.createdAt,color='purple')

# %%
dates = []
for i in df.createdAt:
    d = str(i).split('-')
    if d[1] == '08':
        dates.append(str(i))
    elif (d[1] == '07' or d[1] == '09'):
        d[1] = d[1].split(' ')
        if(int(d[1][0]) >=15 and int(d[1][0]) <= 31):
            dates.append(str(i))

# %%


# %%
print(df.status_id.unique())

# %%
stat = {}
for i in df.status_id:
    if i not in stat:
        stat[i] = 1
    else:
        stat[i]+=1
print(stat)
plt.figure(figsize = (8,8))
plt.pie(stat.values(),labels = stat.keys(),autopct = "%.2f")

# %%
print("The status IDs : ",stat)
plt.figure(figsize = (7,6))
plt.title("Status ID bar chart")
plt.grid(True)
plt.xlabel("Status IDs")
plt.ylim(0,7000)
plt.bar(stat.keys(), height=stat.values(), color = 'darkblue')

# %%
df1 = df[['category','status_id']]
#print(df1)
#for i in df1.categor
cat = list(df.category)
stat = list(df.status_id)
res = {}
for i in stat:
    if i==3:
        ct = cat[stat.index(i)]
        if ct not in res:
            res[ct] = 1
        else:
            res[ct]+=1
print(res)
plt.figure(figsize = (5,5))
plt.title("Categories of complaints resolved")
plt.pie(res.values(),labels = res.keys(),autopct = '%.2f')

# %%
dates = []
for i in df.createdAt:
    d1 = str(i).split(' ')
    d = d1[0].split('-')
    if d[1] == '07':
        dates.append(d1[0])
    elif (d[1] == '08'):
        #d[1] = d[1].split(' ')
        if(int(d[2]) <=5):
            dates.append(d1[0])
    elif(d[1]=='10' and d[2]=='02'):
        dates.append(d1[0])
date = {}
for i in dates:
    if i not in date:
        date[i] = 1
    else:
        date[i]+=1
print(date['2019-10-02'])
plt.figure(figsize = (50,16))
plt.grid(True)
plt.ylim(0,300)
plt.plot_date(date.keys(),date.values())
#plt.hist(dates)

# %%
plt.plot(date.keys(),date.values())

# %%
