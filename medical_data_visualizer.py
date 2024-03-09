import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda x: 1 if x > 25 else 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'].apply(lambda x: 1 if x > 1 else 0))
df['gluc'] = (df['gluc'].apply(lambda x: 1 if x > 1 else 0))

def is_numeric(obj):
    try:
        np.asarray(obj, dtype=float)
    except ValueError:
        return False
    else:
        return True

def draw_cat_plot():
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])
    df_cat = df_cat.rename(columns={'variable': 'variable', 'value': 'count'})

    fig = sns.catplot(data=df_cat, x='variable', y='count', hue='count', col='cardio', kind='bar').fig

    if not is_numeric(df_cat['count']):
        order = df_cat['count'].value_counts().index
        sns.catplot(data=df_cat, x='variable', y='count', hue='count', col='cardio', kind='bar', order=order).fig

    fig.savefig('catplot.png')
    return fig

# Draw Heat Map
def draw_heat_map():
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]
    
    corr = df_heat.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, mask=mask, fmt='.1f', vmax=0.9, square=True)
    
    fig.savefig('heatmap.png')
    return fig