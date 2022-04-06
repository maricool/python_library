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

def column_statistics(df, feature):
    '''
    Computes useful summary statistics of one pandas column (called feature here)
    TODO: How about df.describe()?
    '''
    d = df[feature]
    print('Feature:', feature)
    print('Number:', len(d))
    print('Sum:', d.sum())
    print('Mean:', d.mean())
    print('Median:', d.median())
    print('Standard deviation:', d.std())
    print()

def feature_triangle(df, target, features):
    '''
    Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal
    df: pandas data frame
    target: sting, name of one column, usually the (discrete) thing you are interested in predicting (e.g., species) 
    features: list of strings corresponding to feature columns (e.g., flipper length, width)
    TODO: Sort x and y ranges
    '''
    density = True
    bins = 'auto'
    sns.set_theme(style='ticks')
    n = len(features)
    _, axs = plt.subplots(n, n, figsize=(10,10), sharex=True, sharey=True)
    i = 0
    for i1, feature1 in enumerate(features):
        for i2, feature2 in enumerate(features):
            i += 1
            if i2 > i1:
                axs[i1, i2].axis('off') # Ignore upper triangle
                continue
            plt.subplot(n, n, i)
            if i1 == i2:
                for thing in list(set(df[target])):
                    q = "%s == '%s'"%(target, thing) # Query to isolate 
                    plt.hist(df.query(q)[feature1], bins=bins, density=density, label=thing)
            else:           
                sns.scatterplot(x=df[feature2], y=df[feature1], 
                                hue=df[target], 
                                style=df[target],
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