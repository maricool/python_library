import matplotlib.pyplot as plt
import seaborn as sns

def data_head(df, comment, verbose=False):
    '''
    Utility function for writing out subset of dataframe with comment
    TODO: This does not seem to be necessary
    '''
    if verbose:
        print(comment)
        print(df.head(15))
        print()

def column_statistics(df, feature, condition=None):
    '''
    Computes useful summary statistics of one pandas column (called feature here)
    TODO: How about df.describe() or df.agg?
    '''
    if condition is None:
        d = df[feature]
    else:
        d = df[feature].loc[condition]
    print('Feature:', feature)
    print('Number:', len(d))
    print('Sum:', d.sum())
    print('Mean:', d.mean())
    print('Median:', d.median())
    print('Standard deviation:', d.std())
    print()

def feature_triangle(df, label, features):
    '''
    Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal
    df: pandas data frame
    label: sting, name of one column, usually the (discrete) label you are interested in predicting (e.g., species) 
    features: list of strings corresponding to feature columns (e.g., petal length, petal width)
    '''
    stat='density'
    bins = 'auto'
    sns.set_theme(style='ticks')
    n = len(features)
    _, axs = plt.subplots(n, n, figsize=(10,10))#, sharex=True, sharey=True) # sharex and sharey do not work because of density vs. scatter dims
    i = 0
    for i1, feature1 in enumerate(features):
        for i2, feature2 in enumerate(features):
            i += 1
            if i2 > i1:
                axs[i1, i2].axis('off') # Ignore upper triangle
                continue
            plt.subplot(n, n, i)
            if i1 == i2:
                sns.histplot(df, x=feature1, hue=df[label], stat=stat, bins=bins, legend=(i1==0), kde=True)
            else:           
                sns.scatterplot(x=df[feature2], y=df[feature1], 
                                hue=df[label], 
                                style=df[label],
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
            plt.tight_layout()