import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def key_distribution(df, keys):
    # count the frequency for each key in the dataset
    key_counts = {key: 0 for key in keys}

    for _, row in df.iterrows():
        key = row['Key']
        if key in key_counts:
            key_counts[key] += 1

    # convert for compatibility with matplotlib
    key_counts_series = pd.Series(key_counts, index=keys)

    norm = plt.Normalize(key_counts_series.min(), key_counts_series.max())
    colors = plt.cm.coolwarm(norm(key_counts_series.values))

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(keys, key_counts_series.values, color=colors)

    # fix color scheme 
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Frequency", rotation=270, labelpad=15)

    # graph structure 
    ax.set_title("Key Distribution in Classical Music")
    ax.set_xlabel("Musical Key")
    ax.set_ylabel("Frequency")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45)

    plt.tight_layout()
    plt.show()

def major_minor_keys(df, major_keys, minor_keys):
    df['Mode'] = df['Key'].apply(lambda x: 'Major' if x in major_keys else 'Minor')

    # create distribution plot
    custom_palette = {"Major": "#FF9550",  
                      "Minor": "#11DABB"}  

    sns.countplot(data=df, x="Mode", palette=custom_palette)
    plt.title("Major vs. Minor Keys in Classical Music", fontsize=16)
    plt.xlabel("Mode", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # map keys for analysis 
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    df = pd.read_csv("classical_keys_analysis.csv")
    key_distribution(df, keys)

    major = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    minor = ['C#', 'D#', 'F#', 'G#', 'A#']

    major_minor_keys(df, major, minor)


