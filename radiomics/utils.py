import pickle
import sklearn
import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

############## Pre-process feature table ###############

def get_correlated_features_to_drop(train_dataset, threshold=0.85):
    """Identify highly correlated features to drop based on Spearman correlation."""

    cor = train_dataset.corr('spearman').abs()
    upper_tri = cor.where(np.triu(np.ones(cor.shape), k=1).astype(bool))
    to_drop = []
    for column in upper_tri.columns:
        for row in upper_tri.columns:
            if upper_tri[column][row] > threshold:
                # Drop the feature with the higher overall correlation
                to_drop.append(column if np.sum(upper_tri[column]) > np.sum(upper_tri[row]) else row)

    return np.unique(to_drop)


def preprocessing_train(train_features):
    """Normalize features, remove low-variance features, and drop correlated features from the training data."""

    mean_std = {}

    # Normalize the features
    for var in train_features.columns:
        temp_mean = train_features[var].mean()
        temp_std = train_features[var].std()
        mean_std[var] = (temp_mean, temp_std)
        train_features[var] = (train_features[var] - temp_mean) / temp_std

    # Remove low-variance features
    selector = VarianceThreshold(threshold=0.01)
    selector.fit(train_features)
    filtered_features = train_features.loc[:, selector.get_support()]

    # Drop highly correlated features
    to_drop = get_correlated_features_to_drop(filtered_features)
    final_train_features = filtered_features.drop(to_drop, axis=1)

    return mean_std, selector, to_drop, final_train_features


def preprocessing_test(test_features, mean_std, selector, to_drop):
    """Apply normalization, variance thresholding, and correlation dropping to the test data."""

    # Normalize the features
    for var in test_features.columns:
        test_features[var] = (test_features[var] - mean_std[var][0]) / mean_std[var][1]
    
    # Apply selector and drop correlated features
    filtered_test = test_features.loc[:, selector.get_support()]
    final_test_features = filtered_test.drop(to_drop, axis=1)

    return final_test_features


############## Generate results ###############

def get_optimal_threshold(true_labels, predictions):
    """Find the optimal threshold for predictions using ROC curve analysis."""

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(true_labels, predictions)
    optimal_idx = np.argmax(tpr - fpr)

    return thresholds[optimal_idx]


def get_results(y_true, y_pred, label, optimal_threshold):
    """Compute AUC, accuracy, precision, recall, and F1-score for the predictions at the optimal threshold."""

    results = {}
    fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
    results["auc"] = sklearn.metrics.auc(fpr, tpr)
    y_pred_binary = (y_pred > optimal_threshold).astype(int)
    
    results["accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred_binary)
    results["precision"] = sklearn.metrics.precision_score(y_true, y_pred_binary)
    results["recall"] = sklearn.metrics.recall_score(y_true, y_pred_binary)
    results["f1_score"] = sklearn.metrics.f1_score(y_true, y_pred_binary)

    # Convert results into a DataFrame for output
    df_results = pd.DataFrame(results, index=[label])

    return df_results


def bootstrap(label, pred, metric_func, nsamples=2000):
    """Bootstrap sampling to estimate confidence intervals for a given metric."""

    stats = []
    for _ in range(nsamples):
        random_sample = np.random.randint(len(label), size=len(label))
        stats.append(metric_func(label[random_sample], pred[random_sample]))

    return stats, np.percentile(stats, [2.5, 97.5])


def nom_den(label, pred, metric_func):
    """Get numerator and denominator for accuracy, precision, recall, and F1-score."""

    if metric_func == sklearn.metrics.accuracy_score:
        return np.sum(label == pred), len(pred)
    if metric_func == sklearn.metrics.precision_score:
        return np.sum(pred[label == 1]), np.sum(pred)
    if metric_func == sklearn.metrics.recall_score:
        return np.sum(pred[label == 1]), np.sum(label)
    
    return 0, 0  # Default for unsupported metrics


def get_ci(label, pred, metric_func):
    """Compute confidence interval for a given metric."""

    stats, ci = bootstrap(label, pred, metric_func)
    n, d = nom_den(label, pred, metric_func)

    return stats, f"{n}/{d} ({metric_func(label, pred) * 100:.2f}%) CI [{ci[0]:.2f}, {ci[1]:.2f}]"


def get_ci_for_auc(label, pred, nsamples=2000):
    """Bootstrap to compute confidence interval for AUC."""

    auc_values = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(nsamples):
        idx = np.random.randint(len(label), size=len(label))
        temp_pred = pred[idx]
        temp_fpr, temp_tpr, _ = sklearn.metrics.roc_curve(label[idx], temp_pred)
        auc_values.append(sklearn.metrics.auc(temp_fpr, temp_tpr))
        interp_tpr = np.interp(mean_fpr, temp_fpr, temp_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    ci_auc = np.percentile(auc_values, [2.5, 97.5])

    fpr, tpr, _ = sklearn.metrics.roc_curve(label, pred)

    return auc_values, f"{sklearn.metrics.auc(fpr, tpr):.2f} CI [{ci_auc[0]:.2f}, {ci_auc[1]:.2f}]"


def get_stats_with_ci(y_true, y_pred, label, optimal_threshold):
    """Compute statistics (AUC, accuracy, precision, recall, F1) with confidence intervals."""

    results = {}
    distributions = {}

    distributions["auc"], results["auc"] = get_ci_for_auc(y_true, y_pred)
    y_pred_binary = (y_pred > optimal_threshold).astype(int)

    distributions["accuracy"], results["accuracy"] = get_ci(y_true, y_pred_binary, sklearn.metrics.accuracy_score)
    distributions["precision"], results["precision"] = get_ci(y_true, y_pred_binary, sklearn.metrics.precision_score)
    distributions["specificity"], results["specificity"] = get_ci(1 - y_true, 1 - y_pred_binary, sklearn.metrics.recall_score)
    distributions["recall"], results["recall"] = get_ci(y_true, y_pred_binary, sklearn.metrics.recall_score)
    distributions["f1_score"], results["f1_score"] = get_ci(y_true, y_pred_binary, sklearn.metrics.f1_score)

    df_results = pd.DataFrame(results, index=[label])
    df_distributions = pd.DataFrame(distributions)
    
    return df_distributions, df_results


########### Save and Load Model Parameters ###########

def save_all_params(path, rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train, proba_test, proba_external, data_used="radiomics"):
    """Save model parameters and results to files."""

    pickle.dump(rfe, open(f'{path}rfe_{data_used}.pkl', 'wb'))
    pickle.dump(filtered_col, open(f'{path}filtered_col_{data_used}.pkl', 'wb'))
    pickle.dump(gsearch, open(f'{path}gsearch_{data_used}.pkl', 'wb'))
    pickle.dump([mean_std, selector, to_drop, support], open(f'{path}parameters_{data_used}.pkl', 'wb'))
    pickle.dump(proba_train, open(f'{path}proba_train_{data_used}.pkl', 'wb'))
    pickle.dump(proba_test, open(f'{path}proba_test_{data_used}.pkl', 'wb'))
    pickle.dump(proba_external, open(f'{path}proba_external_{data_used}.pkl', 'wb'))

    return "done"


def load_all_params(path, data_used="radiomics"):
    """Load model parameters and results from files."""

    rfe = pickle.load(open(f'{path}rfe_{data_used}.pkl', 'rb'))
    filtered_col = pickle.load(open(f'{path}filtered_col_{data_used}.pkl', 'rb'))
    gsearch = pickle.load(open(f'{path}gsearch_{data_used}.pkl', 'rb'))
    mean_std, selector, to_drop, support = pickle.load(open(f'{path}parameters_{data_used}.pkl', 'rb'))
    proba_train = pickle.load(open(f'{path}proba_train_{data_used}.pkl', 'rb'))
    
    return rfe, filtered_col, gsearch, mean_std, to_drop, selector, support, proba_train,
