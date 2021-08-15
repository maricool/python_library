import numpy as np
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

# def feature_triangle(df, label, features):
#     '''
#     Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal
#     df: pandas data frame
#     label: sting, name of one column, usually the (discrete) label you are interested in predicting (e.g., species) 
#     features: list of strings corresponding to feature columns (e.g., petal length, petal width)
#     '''
#     stat='density'
#     bins = 'auto'
#     sns.set_theme(style='ticks')
#     n = len(features)
#     _, axs = plt.subplots(n, n, figsize=(10,10))#, sharex=True, sharey=True) # sharex and sharey do not work because of density vs. scatter dims
#     i = 0
#     for i1, feature1 in enumerate(features):
#         for i2, feature2 in enumerate(features):
#             i += 1
#             if i2 > i1:
#                 axs[i1, i2].axis('off') # Ignore upper triangle 
#                 continue
#             plt.subplot(n, n, i)
#             if i1 == i2:
#                 sns.histplot(df, x=feature1, hue=df[label], stat=stat, bins=bins, legend=(i1==0), kde=True)
#             else:           
#                 sns.scatterplot(x=df[feature2], y=df[feature1], 
#                                 hue=df[label], 
#                                 style=df[label],
#                                 legend=None,
#                             )
#             if i1 == len(features)-1: 
#                 plt.xlabel(feature2)
#             else:
#                 plt.xlabel(None)
#                 plt.tick_params(axis='x', which='major', bottom=False, labelbottom=False)
#             if (i2 == 0) and (i1 != 0):
#                 plt.ylabel(feature1)
#             else:
#                 plt.ylabel(None)
#                 plt.tick_params(axis='y', which='major', left=False, labelleft=False)
#             plt.tight_layout()

def feature_triangle(df, label, features, continuous_label=False, 
                                          histograms=True, 
                                          kde=True, 
                                          alpha=1., 
                                          figsize=(10,10), 
                                          jitter=False
                                        ):
    '''
    Triangle plot of list of features split by some characteristic. Histogram distributions along diagonal, correlations off diagonal.
    @params
        df - pandas data frame
        label - string, name of one column, usually the (discrete) label you are interested in predicting (e.g., species) 
        features - list of strings corresponding to feature columns (e.g., petal length, petal width)
    '''

    sns.set_theme(style='ticks')
    fsig = 0.2
    if histograms:
        n = len(features)
    else:
        n = len(features)-1
        raise ValueError('Plotting without histograms does not work yet')

    # Figure out the hue of bars/points
    if label is None:
        hue_hist = None
        hue_scat = None
    else:
        if continuous_label:
            hue_hist = None
        else:
            hue_hist = df[label]
        hue_scat = df[label]

    # Two functions for the crude implementation of jitter
    def add_jitter(values, std):
        return values+np.random.normal(0., std, values.shape)
    def calculate_minimum_difference(series):
        arr = series.to_numpy()
        b = np.diff(np.sort(arr))
        return b[b>0].min()

    # Make the big plot
    _, axs = plt.subplots(n, n, figsize=figsize)
    i = 0
    for i1, feature1 in enumerate(features):
        for i2, feature2 in enumerate(features):
            i += 1
            if i2 > i1:
                if histograms:
                    axs[i1, i2].axis('off') # Ignore upper triangle
                else:
                    axs[i1+1, i2+1].axis('off')
                continue
            plt.subplot(n, n, i)
            if histograms and (i1 == i2):
                sns.histplot(df, x=feature1, 
                                 hue=hue_hist, 
                                 stat='density', 
                                 bins='auto', 
                                 legend=(i1==0), 
                                 kde=kde,
                                )
            else:
                if jitter:
                    std1 = fsig*calculate_minimum_difference(df[feature1])
                    std2 = fsig*calculate_minimum_difference(df[feature2])
                    xdata = add_jitter(df[feature2],std2)
                    ydata = add_jitter(df[feature1],std1)
                else:
                    xdata = df[feature2]
                    ydata = df[feature1]
                sns.scatterplot(x=xdata, y=ydata, 
                                hue=hue_scat, 
                                alpha=alpha,
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
    return plt

# Correlation matrix
def correlation_matrix(df, columns, figsize=(7,7), mask_diagonal=True, mask_upper_triangle=True):
    '''
    Create a plot of the correlation matrix for (continous) columns (features) of dataframe (df)
    '''
    # Calculate correlation coefficients
    corr = df[columns].corr() 
    if mask_diagonal and mask_upper_triangle:
        corr.drop(labels=columns[0], axis=0, inplace=True)  # Remove first row
        corr.drop(labels=columns[-1], axis=1, inplace=True) # Remove last column

    # Create mask
    mask = np.zeros_like(corr, dtype=bool) 
    if mask_upper_triangle and mask_diagonal:
        mask[np.triu_indices_from(mask, k=1)] = True # k=1 does diagonal offset from centre
    elif mask_upper_triangle:
        mask[np.triu_indices_from(mask, k=1)] = True
    elif mask_diagonal:
        mask[np.diag_indices_from(mask)] = True

    # Make the plot
    plt.style.use('seaborn-white') 
    plt.figure(figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    g = sns.heatmap(corr, vmin=-1., vmax=1., cmap=cmap, mask=mask, linewidths=.5,
                    annot=True,
                    square=True,
                    cbar=False,
                )
    g.set_yticklabels(labels=g.get_yticklabels(), va='center') # Centre y-axis ticks
    return plt

