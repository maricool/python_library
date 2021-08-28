import numpy as np
from scipy.stats import randint as sp_randint
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV#, GridSearchCV
from sklearn.metrics import classification_report

def TrainRandomForest(df, target, features, test_size=0.2, random_state=42,
        n_iter=100, cv=3, n_jobs=-1, scoring='recall', verbose=True,
        n_trees_min=1, n_trees_max=1000, 
        max_features_min=1, max_features_max=10, 
        min_samples_split_min=2, min_samples_split_max=100,
        bootstrap=[True, False], 
        ):

    # Create the training and test split (reporducible via the random_state)
    # TODO: Incorporate stratified sampling
    y = df[target]
    X = df[features]
    if test_size is None:
        X_train, y_train = X, y
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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
                                        cv=cv, n_jobs=n_jobs, scoring=scoring)
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

def plotFeatureImportance(best_model, df_features, figsize=(20,10), color='red'):

    # Look at the feature importance to get some insight about how our model is using features
    _, ax = plt.subplots(figsize=figsize)
    ax.bar

    # Calculate importances and sort indices
    importances = best_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    indices = np.argsort(importances)

    # Plot the feature importances of the forest
    ax.barh(df_features.columns[indices], importances[indices], color=color, 
        yerr=std[indices], align='center')
    ax.set_xlabel('Fractional importance')
    ax.tick_params()