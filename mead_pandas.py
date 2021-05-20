import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def column_statistics(df, feature):
    d = df[feature]
    print('Feature:', feature)
    print('Number:', len(d))
    print('Sum:', d.sum())
    print('Mean:', d.mean())
    print('Median:', d.median())
    print('Standard deviation:', d.std())
    print()

def feature_triangle(df, disc, features):

    sns.set_theme(style='ticks')

    n = len(features)

    _, axs = plt.subplots(n, n, figsize=(10,10), sharex=True, sharey=True)

    i = 0
    for i1, feature1 in enumerate(features):
        for i2, feature2 in enumerate(features):
            i += 1
            if i2 > i1:
                axs[i1, i2].axis('off')
                continue
            plt.subplot(n, n, i)
            if i1 == i2:
                for thing in list(set(df[disc])):
                    q = "%s == '%s'"%(disc, thing)
                    plt.hist(df.query(q)[feature1], bins='auto', density=True, label=thing)
            else:           
                sns.scatterplot(x=df[feature2], y=df[feature1], 
                                hue=df[disc], 
                                style=df[disc],
                                legend=None,
                            )
            if i1 == len(features)-1: 
                plt.xlabel(feature2)
            else:
                plt.xlabel(None)
                plt.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
            if (i2 == 0) and (i1 != 0):
                plt.ylabel(feature1)
            else:
                plt.ylabel(None)
                plt.tick_params(axis='y', which='major', left=False, labelleft=False)
            if i1 == i2 == 0: plt.legend()
            plt.tight_layout()