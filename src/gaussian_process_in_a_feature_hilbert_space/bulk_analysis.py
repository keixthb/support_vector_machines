import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.gaussian_process.kernels import RBF
from sklearn import preprocessing
import argparse 

def sliding_window(x, w = 1, d = 0):
    N = len(x)
    x=x.reshape(x.size,)
    m = int(np.ceil((N-w)/(w-d)))
    X = np.zeros([w,m])
    for i,j in zip(range(0,N,w-d),range(0,m)):
        X[:,j] = x[i:i + w]
    return X

def plot_graphs(y_tst,mean,std,title):
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

    print(f"training linear classifier jan {filename2020} {filename2021} m={value_for_m}")
    W=1
    M=value_for_m

    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    ONE_THOUSAND_HOURS = 1000

    data_2020_ct_raw=np.loadtxt(filename2020, dtype = np.float64)
    data_2020_ct_raw=data_2020_ct_raw.reshape(data_2020_ct_raw.size,1) 
    data_2021_ct_raw=np.loadtxt(filename2021, dtype = np.float64)
    data_2021_ct_raw=data_2021_ct_raw.reshape(data_2021_ct_raw.size,1) 

    scaler_2020 = preprocessing.StandardScaler().fit(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])
    train_2020 = scaler_2020.transform(data_2020_ct_raw[0:LENGTH_OF_ONE_WEEK_IN_HOURS])

    x_axis_week = np.arange(LENGTH_OF_ONE_WEEK_IN_HOURS - 1)

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

    '''
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
    '''


    return mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw





def train_linear_classifier_july(filename2020, filename2021, value_for_m):
    print(f"training linear classifier july {filename2020} {filename2021} m={value_for_m}")
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

    '''
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
    plt.title(f"{get_txt_location(filename2020)} m={value_for_m} lin july.png")
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_lin_july.png")

    plt.show()
    plt.cla()
    '''

    return mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw



def train_square_exponential_classifier_jan(filename2020, filename2021, value_for_m):
    print(f"training square exponential classifier jan {filename2020} {filename2021} m={value_for_m}")
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

    '''
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
    plt.title(f"{get_txt_location(filename2020)} m={value_for_m} exp jan.png")
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_exp_jan.png")
    plt.cla()
    '''
    return mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw

def train_square_exponential_classifier_july(filename2020, filename2021, value_for_m):
    print(f"training square exponential classifier july {filename2020} {filename2021} m={value_for_m}")

    
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

    '''
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
    plt.title(f"{get_txt_location(filename2020)} m={value_for_m} exp july")
    plt.legend(loc='upper left')
    plt.savefig(f"{get_txt_location(filename2020)}_m_{value_for_m}_exp_july.png")

    plt.show()
    plt.cla()
    '''
    return mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw


def generate_results(region, data_in_2020_to_predict, data_in_2021_to_predict):
    LENGTH_OF_ONE_WEEK_IN_HOURS = 168
    START_INDEX_FOR_JULY = 4367
    END_INDEX_FOR_JULY = 4535

    ONE_THOUSAND_HOURS = 1000


    my_title = f"{region} data"
    fig, axes = plt.subplots(nrows=2, ncols=6, figsize=(18, 12))


    ###############################    jan m1 lin ######################



    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 1)

    axes[0, 0].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[0, 0].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[0, 0].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[0, 0].set_title('linear jan m=1')
    axes[0, 0].set_ylabel('f(t)')
    axes[0, 0].legend(loc='upper left')




    ###############################    jan m6 lin ######################


    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 6)

    axes[0, 1].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[0, 1].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[0, 1].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[0, 1].set_title('linear jan m=1')
    axes[0, 1].set_ylabel('f(t)')
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set_title('linear jan m=6')


    ###############################    jan m12 lin ######################



    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 12)

    axes[0, 2].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[0, 2].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[0, 2].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )


    axes[0, 2].set_title('linear jan m=1')
    axes[0, 2].set_ylabel('f(t)')
    axes[0, 2].legend(loc='upper left')
    axes[0, 2].set_title('linear jan m=6')

    ###############################    july m1 lin ######################

    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 1)

    axes[0, 3].plot(x_axis_week[0: 24], mean_prediction[0: 24], label="Mean Prediction",color="blue")
    axes[0, 3].plot(x_axis_week[0: 24],data_2021_ct_raw[START_INDEX_FOR_JULY: START_INDEX_FOR_JULY + 24],label="First Week of july 2021",color="red")
    axes[0, 3].fill_between(
        x_axis_week[0: 24].ravel(),
        mean_prediction[0: 24] - 1.96 * std_prediction[0: 24],
        mean_prediction[0: 24] + 1.96 * std_prediction[0: 24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[0, 3].legend(loc='upper left')
    axes[0, 3].set_title('linear july m=1')


    ###############################    july m6 lin ######################
    
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 6)

    axes[0, 4].plot(x_axis_week[0: 24], mean_prediction[0: 24], label="Mean Prediction",color="blue")
    axes[0, 4].plot(x_axis_week[0: 24],data_2021_ct_raw[START_INDEX_FOR_JULY: START_INDEX_FOR_JULY + 24],label="First Week of july 2021",color="red")
    axes[0, 4].fill_between(
        x_axis_week[0: 24].ravel(),
        mean_prediction[0: 24] - 1.96 * std_prediction[0: 24],
        mean_prediction[0: 24] + 1.96 * std_prediction[0: 24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[0, 4].legend(loc='upper left')
    axes[0, 4].set_title('linear july m=6')



    ###############################    july m12 lin ######################

    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_linear_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 12)

    axes[0, 5].plot(x_axis_week[0: 24], mean_prediction[0: 24], label="Mean Prediction",color="blue")
    axes[0, 5].plot(x_axis_week[0: 24],data_2021_ct_raw[START_INDEX_FOR_JULY: START_INDEX_FOR_JULY + 24],label="First Week of july 2021",color="red")
    axes[0, 5].fill_between(
        x_axis_week[0: 24].ravel(),
        mean_prediction[0: 24] - 1.96 * std_prediction[0: 24],
        mean_prediction[0: 24] + 1.96 * std_prediction[0: 24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[0, 5].legend(loc='upper left')
    axes[0, 5].set_title('linear july m=12')

    
    ###############################    jan m1 exp ######################

    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 1)

    axes[1, 0].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 0].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[1, 0].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 0].legend(loc='upper left')
    axes[1, 0].set_title('exp jan m=1')




    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('f(t)')


    ###############################    jan m6 exp ######################
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 6)

    axes[1, 1].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 1].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[1, 1].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 1].legend(loc='upper left')
    axes[1, 1].set_title('exp jan m=6')
    axes[1, 1].set_xlabel('t')

    ###############################    jan m12 exp ######################
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_jan(data_in_2020_to_predict, data_in_2021_to_predict, 12)

    axes[1, 2].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 2].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of 2021",color="red")
    axes[1, 2].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 2].legend(loc='upper left')
    axes[1, 2].set_title('exp jan m=12')
    axes[1, 2].set_xlabel('t')

    ###############################    july m1 exp ######################
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 1)

    axes[1, 3].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 3].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of July 2021",color="red")
    axes[1, 3].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 3].legend(loc='upper left')
    axes[1, 3].set_title('exp jan m=1')
    axes[1, 3].set_xlabel('t')


    ###############################    july m6 exp ######################
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 6)

    axes[1, 4].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 4].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of July 2021",color="red")
    axes[1, 4].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 4].legend(loc='upper left')
    axes[1, 4].set_title('exp jan m=6')
    axes[1, 4].set_xlabel('t')

    ###############################    july m12 exp ######################
    mean_prediction, std_prediction, x_axis_week, data_2021_ct_raw = train_square_exponential_classifier_july(data_in_2020_to_predict, data_in_2021_to_predict, 12)

    axes[1, 5].plot(x_axis_week[0:24], mean_prediction[0:24], label="Mean Prediction",color="blue")
    axes[1, 5].plot(x_axis_week[0:24],data_2021_ct_raw[0:24],label="First Week of July 2021",color="red")
    axes[1, 5].fill_between(
        x_axis_week[0:24].ravel(),
        mean_prediction[0:24] - 1.96 * std_prediction[0:24],
        mean_prediction[0:24] + 1.96 * std_prediction[0:24],
        alpha=0.5,
        label=r"95% Confidence Interval",
        color="teal"
    )

    axes[1, 5].legend(loc='upper left')
    axes[1, 5].set_title('exp jan m=12')
    axes[1, 5].set_xlabel('t')

    plt.suptitle(my_title, fontsize=16)


    plt.savefig(f"{region}.png")
    plt.show()
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

    for data_in_2020_to_predict in list_of_2020_regions_to_analyze:
        data_in_2021_to_predict = get_2021_file_name_from_2020_file(data_in_2020_to_predict)
        current_region = get_txt_location(data_in_2020_to_predict)
        generate_results(current_region, data_in_2020_to_predict, data_in_2021_to_predict)

    return


if(__name__ == "__main__"):
    main()


