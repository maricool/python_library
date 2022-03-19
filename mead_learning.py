import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def TrainRandomForest(df, target, features, test_size=0.2, 
        random_state_split=None, random_state_search=None,
        n_iter=100, cv=3, n_jobs=-1, scoring='accuracy', verbose=True,
        n_trees_min=1, n_trees_max=1000, 
        max_features_min=1, max_features_max=10, 
        min_samples_split_min=2, min_samples_split_max=100,
        bootstrap=[True, False], 
        ):
    '''
    Take a dataframe and train a random forest to predict the (binary) target
    Params:
        df: pandas dataframe
        target: column name of binary target
        features: list of column names of features to use
        test_size: fraction of input to use as test
        random_state: random seed for test-train split
        n_iter: number of forests to generate
        cv: folding for cross validation
        n_jobs: ?
        scoring: scikit learn scoring function 
            https://scikit-learn.org/stable/modules/model_evaluation.html
        varbose: verbosity
        n_trees_min/max: Number of trees in forest
        max_features_min/max: ?
        min_samples_split_min/max: ?
        bootstrap: boolean for bootstrap resampling or not
    '''
    from scipy.stats import randint as sp_randint
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, RandomizedSearchCV#, GridSearchCV
    from sklearn.metrics import classification_report

    # Create the training and test split (reporducible via the random_state)
    # TODO: Incorporate stratified sampling
    y = df[target]
    X = df[features]
    if test_size is None:
        X_train, y_train = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                            random_state=random_state_split)

    # Parameters to sample in random search
    n_trees = sp_randint(n_trees_min, n_trees_max)
    max_features = sp_randint(max_features_min, max_features_max)
    min_samples_split = sp_randint(min_samples_split_min, min_samples_split_max)

    # Create the random search grid
    search_params = {
        'n_estimators': n_trees,
        'max_features': max_features,
        'min_samples_split': min_samples_split,
        'bootstrap': bootstrap
        }

    # Initalise the grid search
    # TODO: What does n_jobs do?
    grid_search = RandomizedSearchCV(RandomForestClassifier(), n_iter=n_iter, 
                                        param_distributions=search_params, 
                                        cv=cv, n_jobs=n_jobs, scoring=scoring,
                                        random_state=random_state_search,
                                    )
    grid_search.fit(X_train, y_train) # This is the time-consuming step

    # Write information to screen
    if verbose:
        print('The best parameters to use are \n', grid_search.best_params_)
        print('Which gives a best cross validation score of', grid_search.best_score_)

    # Extract the useful stuff
    best_params = grid_search.best_params_
    random_forest = RandomForestClassifier(**best_params)
    best_model = grid_search.best_estimator_
    if test_size is not None:
        y_pred = best_model.predict(X_test)
        print(classification_report(y_test, y_pred))
    return random_forest, best_model

def plotFeatureImportance(model, df_features, figsize=(10,5), color='red', alpha=1.):
    '''
    Shamelessly stolen from the S2DS Titanic tutorial
    Make a barplot of the feature importances
    '''

    # Look at the feature importance to get some insight about how our model is using features
    _, ax = plt.subplots(figsize=figsize)
    ax.bar

    # Calculate importance and sort indices
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    idx = np.argsort(importances)

    # Plot the feature importances
    ax.barh(df_features.columns[idx], importances[idx], color=color, alpha=alpha, yerr=std[idx], align='center')
    ax.set_xlabel('Fractional importance')
    ax.tick_params()

def checktraintest(X, y, model, ntrials=5, test_size=0.2):
    '''
    Shamelessly stolen from Viviana Acquaviva
    Evaluates the difference between a classifier's train and test scores 
    in a "k-fold-y" fashion. Output means and std to help determine if 
    the difference is statistically significant.
    '''
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    scores_train = np.zeros(ntrials)
    scores_test = np.zeros(ntrials)

    for i in range(ntrials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        model.fit(X_train, y_train)
        pred_test = model.predict(X_test)
        pred_train = model.predict(X_train)

        scores_test[i] = (accuracy_score(y_test, pred_test))
        scores_train[i] =(accuracy_score(y_train, pred_train))

    print('Training scores '+str(scores_train.mean())+' +- '+str(scores_train.std()))
    print('Test scores '+str(scores_test.mean())+' +- '+str(scores_test.std()))

def stacked_barplot(data, x, y, hue, normalize=False, hue_order=None, **kwargs):
    '''
    Create a stacked barplot, rather than the standard side-by-side seaborn barplot
    data - pandas data frame
        x - column to use for x-axis (continous data)
        y - column to use for y-axis (continuous data)
        hue - column to use for color of the bars (catagorical data)
        normalize - Should the barplot be normalized to sum to unity?
        hue_order - List for the order of the bars in the key
        **kwargs - for df.plot function
    '''
    # First get the data into the correct format using pivot
    dpiv = data.pivot(index=x, columns=hue, values=y)
    if normalize: dpiv = dpiv.div(dpiv.sum(axis='columns'), axis='index')
    if hue_order is not None:
        dpiv = dpiv[hue_order]
    dpiv.plot(kind='bar', stacked=True, **kwargs)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Shamelessly stolen from Viviana Acquaviva
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    if not normalize:
        for i, thing in enumerate(classes):
            print('True number of '+thing+':', np.sum(cm[i,:]))

    # Plot the matrix
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add test
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color='white' if cm[i, j]>thresh else 'black')

    # Finalize
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_confusion_matrix2(y_test, y_pred, labels=None, figsize=(11, 5)):
    '''
    Plot a confusion matrix nicely
    '''
    from sklearn.metrics import confusion_matrix
    plt.subplots(1, 2, figsize=figsize)
    for i, (norm, fmt) in enumerate(zip(['true', None], ['.0%', 'd'])):
        plt.subplot(1, 2, i+1)
        confusion = confusion_matrix(y_test, y_pred, normalize=norm)
        g = sns.heatmap(confusion, annot=True, cbar=False, cmap=plt.cm.Blues, fmt=fmt,
            xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        g.set_yticklabels(labels=g.get_yticklabels(), va='center') 
    plt.show()

def plot_ROC_curve(FPR, TPR, ROC_AUC):
    '''
    Make a plot of the reciever-operator characteristic curve
    '''
    plt.plot(FPR, TPR, lw=2, label='AUC = %0.3f'%(ROC_AUC))
    plt.plot([0., 1.], [0., 1.], color='black', ls=':', label='Chance')
    plt.fill_between(FPR, TPR, alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xlim((0.,1.))
    plt.ylim((0.,1.))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

def plot_decision_tree(model, feature_names, class_names):
    '''
    Make a fancy plot of a decision tree
    NOTE: plot_tree(clf, filled=True) with sklearn.tree.plot_tree is also okay
    @params
        model - should be a fitted instance of DecisionTreeClassifier
        feature_names - List of feature names (often X.columns)
        class_names - List of class names (often y.unique())
    '''
    from six import StringIO
    from sklearn.tree import export_graphviz
    from pydotplus import graph_from_dot_data
    from IPython.display import display, Image
    dot_data = StringIO()
    export_graphviz(
                model,
                out_file=dot_data,
                feature_names=feature_names,
                class_names=class_names,
                filled=True,
                rounded=True,
                )
    graph = graph_from_dot_data(dot_data.getvalue())
    display(Image(graph.create_png())) # display seems necessary here

def plot_decision_boundary(clf, df, features, target, n=(129, 129), alpha=0.2):
    '''
    Plot the decision boundary of a decision classifier
    TODO: Fix colormap for boundary regions the same as for scatterplot
    @params
        clf - Instance of fitted decision classifier
        df - Dataframe
        features - List of two features to use for x, y (e.g., ('x1', 'x2'))
        target - Name of target classification variable (e.g., 'species')
        n - Tuple of pixels for x and y
        alpha - alpha of decision region
    '''
    _, ax = plt.subplots() # Initialise plot and make scatter
    sns.scatterplot(data=df, x=features[0], y=features[1], hue=target)

    # Create a mesh and evaluate classifier across the mesh
    (xmin, xmax) = ax.get_xlim(); (ymin, ymax) = ax.get_ylim()
    x = np.linspace(xmin, xmax, n[0])
    y = np.linspace(ymin, ymax, n[1])
    xs, ys = np.meshgrid(x, y)
    zs = clf.predict(np.c_[xs.ravel(), ys.ravel()])

    # Translate from classifier labels (standard output) to integers
    transdict = {}; classes = df[target].unique()
    for i, item in enumerate(classes):#enumerate(classes):
        transdict[item] = i
    zs = [transdict[z] for z in zs]
    zs = np.array(zs).reshape(xs.shape)

    # Plot bounding region
    #plt.pcolormesh(xs, ys, zs, alpha=alpha, shading='auto')
    plt.contourf(xs, ys, zs, alpha=alpha, levels=len(classes)-1) # Much faster than pcolormesh