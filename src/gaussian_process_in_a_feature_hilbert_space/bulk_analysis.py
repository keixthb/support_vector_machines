import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import argparse 

def sliding_window(x, w = 1, d = 0):
    # This function creates a data array formatted in sliding windows.
    # Parameters: X, the input data, which must be a time series in an array of a singel dimension
    #             w, the width of the window
    #             d, the number of elements of the previous window present in the next one.
    #             For the present homework, we use a window shift of 1 position, so then d = w-1
    #             X is the output, and it is an array with the formatted data.
    N = len(x)
    x=x.reshape(x.size,)
    m = int(np.ceil((N-w)/(w-d)))
    X = np.zeros([w,m])
    for i,j in zip(range(0,N,w-d),range(0,m)):
        X[:,j] = x[i:i + w]
    return X

def plot_graphs(y_tst,mean,std,title):
    # This plots the results of the GP prediction, so you do not spend time figuring this out.
    # Parameters: y_tst is the time series of true regressors
    #             mean is the mean value of the prediction
    #             std is the standard deviation of the prediction
    #             tittle is a string with the title that you want to show in the graphic.
    plt.plot(y_tst_raw[:L,], label="Observations",color="black")
    plt.plot(mean_raw, label="Mean prediction",color="red")
    plt.fill_between(
        np.arange(y_tst[:L,].size).ravel(),
        mean_raw - 1.96 * std_raw,
        mean_raw + 1.96 * std_raw,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("$f(t)$")
    plt.title(title)
    plt.show()

def display_normalized_data(my_training_data, my_testing_data):
    plt.cla()
    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS)


    plt.plot(x_axis_week, my_training_data, label="training data",color="blue")
    plt.plot(x_axis_week, my_testing_data,color="black",label="testing data")
    plt.legend()
    return


def get_txt_files(directory):
    try:

        all_files = os.listdir(directory)

        txt_files = [file for file in all_files if file.endswith('.txt')]

        txt_file_paths = [os.path.join(directory, file) for file in txt_files]

        return txt_file_paths

    except FileNotFoundError:
        print(f"Error: Directory '{directory}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def get_txt_year(my_txt_file):
    my_txt_year = my_txt_file.split("/")[1].split("_")[0]
    return(int(my_txt_year))

def get_txt_location(my_txt_file):
    my_txt_location = my_txt_file.split("/")[1].split("_")[1].split(".")[0]
    return my_txt_location

def get_2021_file_name_from_2020_file(file_name):
    location = get_txt_location(file_name)
    path = file_name.split("/")[0]
    my_file_name = f"{path}/{2021}_{location}.txt"
    return my_file_name





def train_linear_classifier_jan(filename2020, filename2021, value_for_m):

    print(f"train_linear_classifier_jan {filename2020} {filename2021} m={value_for_m}")
    W=1
    M=value_for_m



    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    ONE_THOUSAND_HOURS = 1000



    data_2020_ct_raw=np.loadtxt(filename2020, dtype = np.float64)
    data_2020_ct_raw=data_2020_ct_raw.reshape(data_2020_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimensions for the normalization function to accept it.

    data_2021_ct_raw=np.loadtxt(filename2021, dtype = np.float64)
    data_2021_ct_raw=data_2021_ct_raw.reshape(data_2021_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimension

    scaler_2020 = preprocessing.StandardScaler().fit(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])
    train_2020 = scaler_2020.transform(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])

    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS - 1)
    #print(x_axis_week.shape)

    ft_train = data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS].flatten()
    ft_test = data_2021_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS].flatten()
    t_train = x_axis_week

    X_train = sliding_window(ft_train[:-M],W,W-1).T
    y_train = ft_train[M+W:]

    X_test = sliding_window(ft_test[:-M],W,W-1).T
    y_test = ft_test[M+W:]

    kernel = 1*DotProduct(0.1) + WhiteKernel(0.1)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel

    mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
    plt.plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    plt.plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")

    plt.fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    plt.legend(loc='upper left')
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_linear_jan.png")
    plt.cla()
    return


def train_square_exponential_classifier_jan(filename2020, filename2021, value_for_m):
    print(f"train_square_exponential_classifier_jan {filename2020} {filename2021} m={value_for_m}")
    W=1
    M=value_for_m



    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    ONE_THOUSAND_HOURS = 1000



    data_2020_ct_raw=np.loadtxt(filename2020, dtype = np.float64)
    data_2020_ct_raw=data_2020_ct_raw.reshape(data_2020_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimensions for the normalization function to accept it.

    data_2021_ct_raw=np.loadtxt(filename2021, dtype = np.float64)
    data_2021_ct_raw=data_2021_ct_raw.reshape(data_2021_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimension

    scaler_2020 = preprocessing.StandardScaler().fit(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])
    train_2020 = scaler_2020.transform(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])

    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS - 1)
    #print(x_axis_week.shape)

    ft_train = data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS].flatten()
    ft_test = data_2021_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS].flatten()
    t_train = x_axis_week

    X_train = sliding_window(ft_train[:-M],W,W-1).T
    y_train = ft_train[M+W:]

    X_test = sliding_window(ft_test[:-M],W,W-1).T
    y_test = ft_test[M+W:]

    kernel = 2 * RBF(length_scale = 2.0, length_scale_bounds = (1e-2, 1e2)) + WhiteKernel(0.1)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel

    mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
    plt.plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    plt.plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")

    plt.fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    plt.legend(loc='upper left')
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_exp_jan.png")
    plt.cla()
    return


def train_linear_classifier_july(filename2020, filename2021, value_for_m):
    print(f"train_linear_classifier_july {filename2020} {filename2021} m={value_for_m}")
    W=1
    M=value_for_m



    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    START_INDEX_FOR_JULY = 4367
    END_INDEX_FOR_JULY = 4535

    ONE_THOUSAND_HOURS = 1000

    data_2020_ct_raw=np.loadtxt(filename2020, dtype = np.float64)
    data_2020_ct_raw=data_2020_ct_raw.reshape(data_2020_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimensions for the normalization function to accept it.

    data_2021_ct_raw=np.loadtxt(filename2021, dtype = np.float64)
    data_2021_ct_raw=data_2021_ct_raw.reshape(data_2021_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimension

    scaler_2020 = preprocessing.StandardScaler().fit(data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY])
    train_2020 = scaler_2020.transform(data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY])

    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS - 1)
    #print(x_axis_week.shape)

    ft_train = data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY].flatten()
    ft_test = data_2021_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY].flatten()
    t_train = x_axis_week

    X_train = sliding_window(ft_train[:-M],W,W-1).T
    y_train = ft_train[M+W:]

    X_test = sliding_window(ft_test[:-M],W,W-1).T
    y_test = ft_test[M+W:]

    kernel = 1*DotProduct(0.1) + WhiteKernel(0.1)
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel

    mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
    plt.plot(x_axis_week[0: 24], mean_prediction[0: 24], label="Mean Prediction",color="blue")
    plt.plot(x_axis_week[0: 24],data_2021_ct_raw[START_INDEX_FOR_JULY: START_INDEX_FOR_JULY + 24],label="First Week of 2021",color="red")

    plt.fill_between(
        x_axis_week[0: 24].ravel(),
        mean_prediction[0: 24] - 1.96 * std_prediction[0: 24],
        mean_prediction[0: 24] + 1.96 * std_prediction[0: 24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    plt.legend(loc='upper left')
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_lin_july.png")

    plt.show()
    plt.cla()
    return


def train_square_exponential_classifier_july(filename2020, filename2021, value_for_m):
    print(f"train_square_exponential_classifier_july {filename2020} {filename2021} m={value_for_m}")

    
    W=1
    M=value_for_m



    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    START_INDEX_FOR_JULY = 4367
    END_INDEX_FOR_JULY = 4535

    ONE_THOUSAND_HOURS = 1000

    data_2020_ct_raw=np.loadtxt(filename2020, dtype = np.float64)
    data_2020_ct_raw=data_2020_ct_raw.reshape(data_2020_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimensions for the normalization function to accept it.

    data_2021_ct_raw=np.loadtxt(filename2021, dtype = np.float64)
    data_2021_ct_raw=data_2021_ct_raw.reshape(data_2021_ct_raw.size,1) # Here we need to convert it to an array with (N,1) dimension

    scaler_2020 = preprocessing.StandardScaler().fit(data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY])
    train_2020 = scaler_2020.transform(data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY])

    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS - 1)
    #print(x_axis_week.shape)

    ft_train = data_2020_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY].flatten()
    ft_test = data_2021_ct_raw[START_INDEX_FOR_JULY:END_INDEX_FOR_JULY].flatten()
    t_train = x_axis_week

    X_train = sliding_window(ft_train[:-M],W,W-1).T
    y_train = ft_train[M+W:]

    X_test = sliding_window(ft_test[:-M],W,W-1).T
    y_test = ft_test[M+W:]

    kernel = 2 * RBF(length_scale = 2.0, length_scale_bounds = (1e-2, 1e2)) + WhiteKernel(0.1)


    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gaussian_process.fit(X_train, y_train)
    gaussian_process.kernel

    mean_prediction, std_prediction = gaussian_process.predict(X_test, return_std=True)
    plt.plot(x_axis_week[0: 24], mean_prediction[0: 24], label="Mean Prediction",color="blue")
    plt.plot(x_axis_week[0: 24],data_2021_ct_raw[START_INDEX_FOR_JULY: START_INDEX_FOR_JULY + 24],label="First Week of 2021",color="red")

    plt.fill_between(
        x_axis_week[0: 24].ravel(),
        mean_prediction[0: 24] - 1.96 * std_prediction[0: 24],
        mean_prediction[0: 24] + 1.96 * std_prediction[0: 24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    plt.legend(loc='upper left')
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_exp_july.png")

    plt.show()
    plt.cla()
    return







def main():

    directory_path = 'data/'
    txt_files = get_txt_files(directory_path)
    list_of_2020_regions_to_analyze = []
    if txt_files:
        print(f'Text files in {directory_path}:')
        for txt_file in txt_files:
            year = get_txt_year(txt_file)
            if(year == 2020):
                list_of_2020_regions_to_analyze.append(txt_file)
            
    else:
        print('No text files found.')


    values_for_m = [1,6,12]

    for region_in_2020_to_predict in list_of_2020_regions_to_analyze:
        region_in_2021_to_predict = get_2021_file_name_from_2020_file(region_in_2020_to_predict)
        print(region_in_2021_to_predict)

        for value_of_m in values_for_m:
            #train_linear_classifier_jan(region_in_2020_to_predict, region_in_2021_to_predict, value_of_m)
            #train_square_exponential_classifier_jan(region_in_2020_to_predict, region_in_2021_to_predict, value_of_m)
            #train_linear_classifier_july(region_in_2020_to_predict, region_in_2021_to_predict, value_of_m)
            train_square_exponential_classifier_july(region_in_2020_to_predict, region_in_2021_to_predict, value_of_m)
        


    return


if(__name__ == "__main__"):
    main()
