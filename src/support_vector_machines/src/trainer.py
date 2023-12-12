import numpy as np
import src.parallel as para
import src.data as data
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay




def get_number_of_points(matrix_data):
    return matrix_data.shape[0]

def get_hilbert_space(lower_bound, upper_bound, total_number_of_steps):
    return np.logspace(np.log10(lower_bound), np.log10(upper_bound), total_number_of_steps)


def get_number_of_points_in_margin(matrix_data, vector_labels, value_of_c):
    linear_support_vector_classifier = svm.SVC(kernel="linear", C=value_of_c).fit(matrix_data, vector_labels)
    return (linear_support_vector_classifier.support_vectors_.size // 2)

def check_if_value_of_c_splits_data_evenly(number_of_points_in_margin, total_number_of_points):
    return number_of_points_in_margin == (total_number_of_points // 2)



def optimize(training_data, config_data):

    dm = para.DistributedMemory()


    start = dm.rank * config_data.size_per_process
    end = (dm.rank + 1) * config_data.size_per_process
    total_number_of_steps = dm.size * config_data.size_per_process




    number_of_points = get_number_of_points(training_data.get_matrix_x())
    hilbert_space = get_hilbert_space(config_data.lower_bound, config_data.upper_bound, total_number_of_steps)

    hilbert_space_per_process = [hilbert_space[i] for i in range(start, end)]
    hilbert_space_index_per_process = [i for i in range(start, end)]

    results_per_process = []




    training_error = np.zeros((len(hilbert_space_per_process), 10))
    testing_error = np.zeros((len(hilbert_space_per_process), 10))




    for j, value_of_c in enumerate(hilbert_space_per_process):

        my_classifier = svm.SVC(kernel="linear", C=value_of_c).fit(training_data.get_matrix_x(), training_data.get_vector_y())

        number_of_points_in_margin = get_number_of_points_in_margin(training_data.get_matrix_x(), training_data.get_vector_y(), value_of_c)
        value_of_c_splits_data_evenly = check_if_value_of_c_splits_data_evenly(number_of_points_in_margin, number_of_points)
        if(value_of_c_splits_data_evenly):
            results_per_process.append(value_of_c)


        for i in range(10):
            testing_data = data.Data().configure(config_data).use_static_points()

            testing_predictions = my_classifier.predict(testing_data.get_matrix_x())
            training_predictions = my_classifier.predict(training_data.get_matrix_x())


            training_error[j, i] = np.mean(abs(training_data.get_vector_y() - training_predictions) / 2)
            testing_error[j, i] = np.mean(abs(testing_data.get_vector_y() - testing_predictions) / 2)


    emperical_risk = np.mean(training_error, axis = 1)
    risk = np.mean(testing_error, axis = 1)

    optimal_c = None

    try:
        optimal_c =  sum(results_per_process) / len(results_per_process)
        optimal_c = round(optimal_c, 3)

    except Exception as ex:
        print("No optimal value of c found with given parameters")








    return hilbert_space, optimal_c, risk, emperical_risk



'''

number_of_points = lib.get_number_of_points(matrix_data)
hilbert_space = lib.get_hilbert_space(config_data[2], config_data[1], total_number_of_steps)
hilbert_space_per_process = [hilbert_space[i] for i in range(start, end)]
hilbert_space_index_per_process = [i for i in range(start, end)]


for hilbert_space_index_per_process, value_of_c in zip(hilbert_space_index_per_process, hilbert_space_per_process):
    number_of_points_in_margin = lib.get_number_of_points_in_margin(matrix_data, vector_labels, value_of_c)
    value_of_c_splits_data_evenly = lib.check_if_value_of_c_splits_data_evenly(number_of_points_in_margin, number_of_points)
    if(value_of_c_splits_data_evenly):
        results_per_process.append(value_of_c)
    risk_measurements = lib.calculate_error(value_of_c, matrix_data, vector_labels, config_data)
    risks_per_process.append(risk_measurements[0])
    emperical_risks_per_process.append(risk_measurements[1])
    risks_minus_emperical_risks_per_process.append(risk_measurements[2])
    lib.save_plot_iteration(value_of_c, matrix_data, vector_labels, hilbert_space_index_per_process, number_of_points_in_margin)

'''
