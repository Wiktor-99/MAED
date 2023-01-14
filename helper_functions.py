import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve


def plot_time_line_charts(data_frame, ylabels):
    fig = plt.figure(figsize=(13,10))
    fig.subplots_adjust(hspace=0.15,wspace=0.01)

    n_row = len(ylabels)
    n_col = 1
    for count, ylabel in enumerate(ylabels):
        ax = fig.add_subplot(n_row, n_col, count+1)
        ax.plot(data_frame["created at"].to_numpy(dtype='datetime64'), data_frame[ylabel].to_numpy())
        ax.set_ylabel(ylabel)

def fill_missing_hours_with_zero(data_count_by_hour_dict):
    for i in range(24):
        if i not in data_count_by_hour_dict:
            data_count_by_hour_dict[i] = 0

    data_count_by_hour_list = list(dict(sorted(data_count_by_hour_dict.items())).values())
    return np.array(data_count_by_hour_list).astype(float)

def plot_histogram_of_tweet_during_time(elon_data_count_by_hour, not_elon_data_count_by_hour, y_label, title):
    bar_width = 0.4
    hours = np.arange(24)
    plt.figure(figsize=(20, 10))
    plt.bar(hours - bar_width/2, elon_data_count_by_hour.values, width=bar_width, label='Elon')
    plt.bar(hours + bar_width/2, not_elon_data_count_by_hour, width=0.4, label='Nie Elon')

    plt.xticks(hours, hours)
    plt.xlabel("Godzina")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def add_annotations_columns_to_array(columns, used_annotations_domain):
    for i in range(used_annotations_domain):
        colum = f'context annotations domain {i}'
        columns.extend([colum])

def remove_null_annotations(used_annotations, column_specific_name, df):
    for i in range(used_annotations):
        col = f'context annotations {column_specific_name} {i}'
        df = df[df[col].notnull()]

    return df

def add_columns_to_be_merged(columns_to_be_merged, used_annotations, column_specific_name):
    for i in range(used_annotations):
        col = f'context annotations {column_specific_name} {i}'
        columns_to_be_merged.extend([col])

    return columns_to_be_merged

def map_value(value):
    if value == 0:
        return 0
    return 1

def hot_one_encode_annotation_columns(columns_list, data_frame, map_value):
    frame = pd.DataFrame()
    frame = pd.concat([frame, pd.get_dummies(data_frame[columns_list].astype(str), prefix='', prefix_sep='')], axis=1)
    frame = frame.groupby(by=frame.columns, axis=1).sum()
    frame = frame.applymap(map_value)
    data_frame.drop(columns_list, axis=1, inplace=True)
    return pd.concat([data_frame, frame], axis=1)

def add_annotations_columns_to_be_drooped(used_annotations, column_specific_name, columns):
    annotations = 6
    for i in range(annotations - used_annotations):
        colum = f'context annotations {column_specific_name} {used_annotations + i}'
        columns.extend([colum])

def plot_learning_curve(
    estimator,
    X,
    y,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):


    plt.title('Krzywa uczenia oraz krzywa walidacji')
    plt.xlabel("Liczba próbek")
    plt.ylabel("Dokładność")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    );
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Dokładność uczenia"
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Dokładność walidacji krzyżowej"
    )
    plt.legend(loc="best")
    plt.show()



if __name__ == "__main__":
    pass