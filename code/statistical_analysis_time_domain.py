import timeit
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt


def convert_to_csv(device, signal):
    """
    This function convert all the raw data to csv file and provides proper header and file name accordingly.
    :param device: can be either 'phone' or 'watch'
    :param signal: can be either 'accel' or 'gyro'
    :return: saves raw dataset with header and correct file name as a csv file
    """
    subject_counter = 0
    for subject_ID in range(1600, 1651):
        data = pd.read_csv(f'./raw/{device}/{signal}/data_{subject_ID}_{signal}_{device}.txt', sep=",", header=None)
        data.columns = ["subject_ID", "activity_ID", "Timestamp", f"x_{device}_{signal}", f"y_{device}_{signal}",
                        f"z_{device}_{signal}"]
        saveing_directory = f'{device}_{signal}/S{subject_counter}_{device}_{signal}.csv'
        data.to_csv(saveing_directory)
        subject_counter += 1
        print(subject_counter)


# convert_to_csv(device='watch', signal='gyro')


def zero_crossing(window):
    """
    :param window: specific window of the row dataset that we want to calculate zero_crossing for it.
    :return: an integer representing the zero crossing rate
    """
    file_sign = np.sign(window)
    file_sign[file_sign == 0] = -1
    zero_crossing = np.where(np.diff(file_sign))[0]
    return len(zero_crossing)


def mean_crossing_rate(window):
    """
    :param window: specific window of the row dataset that we want to calculate mean_crossing for it.
    :return: an integer representing the mean crossing rate
    """
    mean_crossing_counter = 0
    mean = window.mean()
    subtraction = window - mean
    file_sign = np.sign(subtraction)
    for i in range(len(file_sign)):
        if (file_sign.iloc[i]).all() == 1:
            mean_crossing_counter += 1
    return mean_crossing_counter


def statistical_feature_extraction(window_size, signal, axis, device, subject_ID):
    """
    1) THis function first load the raw data from directory, if it could not find the directory, it threw an exception
    2) Segments the signal into 10 seconds window sizes
    3) Calculates features min, max, mean, standard deviation, median, variance, zero_crossing, mean_crossing for
     each window and each signal
    4) Saves all the extracted features related to each signal in a csv file.
    :param window_size: must be in seconds
    :param signal: can be either 'accel' or 'gyro'
    :param axis: 'x', 'y', 'z'
    :param device: can be either 'phone' or 'watch'
    :param subject_ID: should be any integer between 0 to 50
    :return:
    """

    start_running = timeit.default_timer()
    try:
        directory = f'data/row_data/{device}_{signal}/S{subject_ID}_{device}_{signal}.csv'
        sampling_rate = 20
        window_size = int(sampling_rate * window_size)
        # print(window_size)
    except:
        print('Error! Can not find such directory.')

    raw_signal = pd.read_csv(directory)
    win_count = 0
    total_win_count = 0
    features_for_all_windows_one_activity = []
    features_for_all_windows_all_activities = []
    column_title = f'{axis}_{device}_{signal}'
    for class_label in np.append(range(1, 14), range(15, 20)):
        activity_ID = chr(class_label + 64)
        raw_data_one_activity = np.array(raw_signal.loc[raw_signal['activity_ID'] == activity_ID, [column_title]])
        raw_data_one_activity = pd.DataFrame(raw_data_one_activity)

        for data_point in range(0, len(raw_data_one_activity), window_size):
            win_count += 1
            start = data_point
            end = start + window_size
            time_domain_window = raw_data_one_activity[start:end]

            time_mean = pd.Series(time_domain_window.mean()).rename(f'{axis}_{signal}_mean')
            time_min = pd.Series(time_domain_window.min()).rename(f'{axis}_{signal}_min')
            time_max = pd.Series(time_domain_window.max()).rename(f'{axis}_{signal}_max')
            time_std = pd.Series(time_domain_window.std()).rename(f'{axis}_{signal}_std')
            time_median = pd.Series(time_domain_window.median()).rename(f'{axis}_{signal}_median')
            time_variance = pd.Series(time_domain_window.var()).rename(f'{axis}_{signal}_variance')
            zero_crossing_rate = pd.Series(zero_crossing(time_domain_window)).rename(
                f'{axis}_{signal}_zero_crossing')
            mean_crossing = pd.Series(mean_crossing_rate(time_domain_window)).rename(
                f'{axis}_{signal}_mean_crossing')
            activity_id_ = pd.Series(activity_ID).rename('Activity_ID')

            features_for_one_window_one_activity = pd.concat(
                [time_mean, time_min, time_max, time_std, time_median, time_variance, zero_crossing_rate, mean_crossing,
                 activity_id_], axis=1)
            features_for_all_windows_one_activity.append(features_for_one_window_one_activity)
            # print(features_for_all_windows)

        print('Window count', win_count)
        total_win_count += win_count
        win_count = 0
        features_for_all_windows_all_activities.append(features_for_all_windows_one_activity)
    features = pd.concat(features_for_all_windows_all_activities[0], ignore_index=False)
    print(features)
    save_as_directory = f'feature_label_tables/feature_{device}_{signal}/feature_S{subject_ID}_{axis}_{device}_{signal}.csv'
    features.to_csv(save_as_directory, encoding='utf-8', index=False)
    finish_running = timeit.default_timer()
    print('Total number of windows: ', total_win_count)
    print('Running time: ', finish_running - start_running)


# statistical_feature_extraction(window_size=10, signal='accel', axis='z', device='phone', subject_ID=0)

def feature_extraction_for_all_subjects():
    device_list = ['phone', 'watch']
    signal_list = ['accel', 'gyro']
    axis_list = ['x', 'y', 'z']

    for device in device_list:
        for signal in signal_list:
            for axis in axis_list:
                for subject_ID in range(0, 51):
                    print('calculating: ', device, signal, axis, subject_ID)
                    print('=====================================================\
                          ========================================================')
                    statistical_feature_extraction(window_size=10, signal=signal, axis=axis, device=device,
                                                   subject_ID=subject_ID)


# feature_extraction_for_all_subjects()


def input_features_labels(device, signal, subject_ID):
    """
    This function prepares the feature matrix and corresponding label vector for the classifiers
    1) drops empty fields
    2) drops two less informative features zero_crossing and mean_crossing
    3) splits the train and test set
    4) finally, normalizes the features with scalar.transform function

    :param device: 'phone' or 'watch'
    :param signal: 'accel' or 'gyro'
    :param subject_ID: int between 0 to 50
    :return: dataframes containing normalized_feature_train, normalized_feature_test, label_train, label_test,
                normalized_all_feature, all_labels
    """

    directory = f'data/feature_label_tables/feature_{device}_{signal}/feature_S{subject_ID}_all_axis_{device}_{signal}'
    data = pd.read_csv(directory)
    data = data.dropna()

    # since all zero_crossing and mean_crossing metrics are zero and 200, respectively,
    # regardless of the signal and the activity, we ignore this feature.
    features = data.drop(columns=[f'x_{signal}_zero_crossing', f'x_{signal}_mean_crossing',
                                  f'y_{signal}_zero_crossing', f'y_{signal}_mean_crossing',
                                  f'z_{signal}_zero_crossing', f'z_{signal}_mean_crossing',
                                  'Activity_ID'])

    all_labels = data[['Activity_ID']]

    feature_train, feature_test, label_train, label_test = train_test_split(
        features, all_labels, test_size=0.2, shuffle=True)
    # feature normalization
    scalar = StandardScaler().fit(feature_train)
    normalized_feature_train = scalar.transform(feature_train)
    normalized_feature_test = scalar.transform(feature_test)
    normalized_all_feature = scalar.transform(features)
    # convert 'numpy.ndarray' to pandas dataframe
    normalized_feature_train = pd.DataFrame(normalized_feature_train)
    normalized_feature_test = pd.DataFrame(normalized_feature_test)
    normalized_all_feature = pd.DataFrame(normalized_all_feature)

    return normalized_feature_train, normalized_feature_test, label_train, label_test, normalized_all_feature, all_labels


# input_features_labels(device='phone', signal='accel', subject_ID=10)


def plotitng_confusion_matrix(conf_matrix, evaluation_mode, subject_ID):
    '''

    :param conf_matrix:confusion_matrix
    :param evaluation_mode: 'personal' for 10-fold cross-validation, 'impersonal' for LOSO cross-validation
    :param subject_ID:int between 0 to 50
    :return:
    '''
    df_cm = pd.DataFrame(conf_matrix, index=[i for i in "ABCDEFGHIJKLMOPQRS"],
                         columns=[i for i in "ABCDEFGHIJKLMOPQRS"])
    plt.figure(figsize=(10, 10))
    sns.heatmap(df_cm, annot=True)
    plt.show()
    # plt.savefig(f'figures/{evaluation_mode}/{subject_ID}.jpg')


# plotitng_confusion_matrix(confusion_matrix)

def accuracy_per_class(conf_matrix, row_index, to_print=True):
    """
    code reference: https://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
    :param conf_matrix: np.array containing the confusion matrix
    :param row_index: must be integer that means which class to choose to do one-vs-the-rest calculation
    :param to_print: if True it will prints the TP, TN, FP and FP measurements
    :return: a float number representing the accuracy
    """
    TP = conf_matrix[row_index, row_index]  # correctly labeled as i
    FN = conf_matrix[:, row_index].sum() - TP  # incorrectly labeled as i
    FP = conf_matrix[row_index, :].sum() - TP  # incorrectly labeled as non-i
    TN = conf_matrix.sum().sum() - TP - FN - FP
    if to_print:
        print('TP: {}'.format(TP))
        print('FN: {}'.format(FN))
        print('FP: {}'.format(FP))
        print('TN: {}'.format(TN))
    accuracy = (TN + TP) / (TP + FN + FP + TN)
    return accuracy


def personal_model_rf(device, signal, subject_ID):
    """
    This function trains a random forest classifier. Then evaluates the model using 10-fold cross-validation
    approach for each individual. Since this model evaluates based on each subject, it is called personal model.

    :param device: can be either 'phone' or 'watch'
    :param signal: can be either 'accel' or 'gyro'
    :param subject_ID: should be any integer between 0 to 50
    :return:
    """
    # getting normalized features and labels for train and test set from "input_features_labels" function
    feature_train, feature_test, label_train, label_test, _, _ = input_features_labels(device=device, signal=signal,
                                                                                       subject_ID=subject_ID)
    label_train = label_train.values.ravel()
    label_test = label_test.values.ravel()

    # classifier configuration
    cross_validation = StratifiedKFold(n_splits=10, shuffle=True, random_state=50)
    rf = RandomForestClassifier(random_state=0)
    hyperparams = {"n_estimators": [30, 50, 100], "max_depth": [10, 30, 50]}
    clf = GridSearchCV(estimator=rf, param_grid=hyperparams, scoring="accuracy", cv=cross_validation, refit=True,
                       verbose=0)

    clf.fit(feature_train, label_train)
    print('Best parameters: ', clf.best_params_)
    prediction = clf.predict(feature_test)
    report = sklearn.metrics.classification_report(label_test, prediction, digits=3, zero_division=1)

    conf_matrix = confusion_matrix(label_test, prediction)
    print(conf_matrix.shape[0])
    # plotitng_confusion_matrix(confusion_matrix=con_matrix, evaluation_mode='personal', subject_ID=subject_ID)
    print(report)
    for row in range(conf_matrix.shape[0]):
        print(f'Accuracy for class {row}: ',
              accuracy_per_class(conf_matrix=conf_matrix, row_index=row, to_print=False))


for subject in range(0, 51):
    print(f'***********************For subject ID ({subject}) ***********************************')
    personal_model_rf(device='phone', signal='accel', subject_ID=subject)
    # personal_model_rf(device='phone', signal='gyro', subject_ID=subject)
    # personal_model_rf(device='watch', signal='accel', subject_ID=subject)
    # personal_model_rf(device='watch', signal='gyro', subject_ID=subject)


def LOSO_cross_validation(signal, device):
    """
    This function evaluate the model based on Leave_one_subject_out (LOSO) cross validation or impersonal model.

    :param device: can be either 'phone' or 'watch'
    :param signal: can be either 'accel' or 'gyro'
    :return:
    """
    rf = RandomForestClassifier(random_state=0)
    hyperparams = {"n_estimators": [30, 50, 100], "max_depth": [10, 30, 50]}
    clf = GridSearchCV(estimator=rf, param_grid=hyperparams, scoring="accuracy", cv=None, refit=True, verbose=0)

    all_subjects_but_one = []
    for subject_out in range(0, 51):
        for subject_in in range(0, 51):
            if subject_in == subject_out:
                print(f'Leaving subject {subject_out} out: \n ==================================================')
            else:
                _, _, _, _, normalized_all_feature, all_labels = input_features_labels(device=device,
                                                                                       signal=signal,
                                                                                       subject_ID=subject_in)
                feature_label = pd.concat([normalized_all_feature, all_labels], axis=1)
                all_subjects_but_one.append(feature_label)

        all_subjects_but_one = pd.concat(all_subjects_but_one, axis=0)
        all_subjects_but_one = all_subjects_but_one.dropna()
        feature_train = all_subjects_but_one.drop(columns=['Activity_ID'])
        label_train = all_subjects_but_one['Activity_ID']
        print(feature_train)
        print(label_train)

        _, _, _, _, feature_test, label_test = input_features_labels(device=device,
                                                                     signal=signal, subject_ID=subject_out)
        print(feature_test)
        print(label_test)
        clf.fit(feature_train, label_train)
        print('Best parameters: ', clf.best_params_)
        prediction = clf.predict(feature_test)
        report = sklearn.metrics.classification_report(label_test, prediction, digits=3, zero_division=1)

        conf_matrix = confusion_matrix(label_test, prediction)
        print(conf_matrix.shape[0])
        # plotitng_confusion_matrix(confusion_matrix=con_matrix, evaluation_mode='personal', subject_ID=subject_ID)
        print(report)
        for row in range(conf_matrix.shape[0]):
            print(f'Accuracy for class {row}: ',
                  accuracy_per_class(conf_matrix=conf_matrix, row_index=row, to_print=False))

        all_subjects_but_one = []


LOSO_cross_validation(signal='accel', device='phone')
LOSO_cross_validation(signal='gyro', device='phone')
LOSO_cross_validation(signal='accel', device='watch')
LOSO_cross_validation(signal='gyro', device='watch')
