import numpy as np
import yaml
import random

np.random.seed(42)


def generate_four_random_labels():
    values = [-1, -1, 1, 1]
    random.shuffle(values)
    return values

def generate_random_coordinate():
    return [round(random.uniform(0.0, 10.0),1), round(random.uniform(0, 10.0),1)]

def generate_four_random_coordinates():
    return [generate_random_coordinate(), generate_random_coordinate(), generate_random_coordinate(), generate_random_coordinate()]


class ConfigData:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.size_per_process = config.get('size_per_process', 64)
        self.upper_bound = config.get('upper_bound', 150)
        self.lower_bound = config.get('lower_bound', 0.011)
        self.coordinates_of_centers_in_r2 = config.get('coordinates_of_centers_in_r2', [])
        self.respective_labels_of_clusters = config.get('respective_labels_of_clusters', [])
        self.standard_deviation_of_clusters = config.get('standard_deviation_of_clusters', 0.075)
        self.number_of_data_samples_per_cluster = config.get('number_of_data_samples_per_cluster', 50)

    def __str__(self):
        return f"ConfigData: \n" \
               f"Size Per Process: {self.size_per_process}\n" \
               f"Upper Bound: {self.upper_bound}\n" \
               f"Lower Bound: {self.lower_bound}\n" \
               f"Coordinates of Centers in R2: {self.coordinates_of_centers_in_r2}\n" \
               f"Respective Labels of Clusters: {self.respective_labels_of_clusters}\n" \
               f"Standard Deviation of Clusters: {self.standard_deviation_of_clusters}\n" \
               f"Number of Data Samples Per Cluster: {self.number_of_data_samples_per_cluster}"



class Data:
    def __init__(self):
        self._matrix_x = None
        self._vector_y = None

        self.coordinates_of_centers_in_r2 = None
        self.respective_labels_of_clusters = None
        self.standard_deviation_of_clusters = None
        self.number_of_data_samples_per_cluster = None

    def configure(self, config_data):
        self.coordinates_of_centers_in_r2 = config_data.coordinates_of_centers_in_r2
        self.respective_labels_of_clusters = config_data.respective_labels_of_clusters
        self.standard_deviation_of_clusters = config_data.standard_deviation_of_clusters
        self.number_of_data_samples_per_cluster = config_data.number_of_data_samples_per_cluster
        return self

    def _set_matrix_x(self, coordinates_of_centers_in_r2, standard_deviation_of_clusters, number_of_data_samples_per_cluster):
        self._matrix_x = np.vstack([np.random.normal(coordinates_of_centers_in_r2[i], standard_deviation_of_clusters, (number_of_data_samples_per_cluster, len(coordinates_of_centers_in_r2[0]))) for i in range(len(coordinates_of_centers_in_r2))])
        return

    def get_matrix_x(self):
        return self._matrix_x

    def _set_vector_y(self, respective_labels_of_clusters, number_of_data_samples_per_cluster):
        self._vector_y = np.hstack([np.full(number_of_data_samples_per_cluster, respective_labels_of_clusters[i]) for i in range(len(respective_labels_of_clusters))])

    def get_vector_y(self):
        return self._vector_y

    def use_random_points(self):
        self._set_matrix_x(generate_four_random_coordinates(), self.standard_deviation_of_clusters, self.number_of_data_samples_per_cluster)
        self._set_vector_y(generate_four_random_labels(), self.number_of_data_samples_per_cluster)
        return self

    def use_static_points(self):
        self._set_matrix_x(self.coordinates_of_centers_in_r2, self.standard_deviation_of_clusters, self.number_of_data_samples_per_cluster)
        self._set_vector_y(self.respective_labels_of_clusters, self.number_of_data_samples_per_cluster)
        return self

    def print_data(self):
        print("Matrix X:")
        print(self._matrix_x)
        print("Vector Y:")
        print(self._vector_y)

    def __str__(self):
        return f"Data instance:\nMatrix X:\n{self._matrix_x}\nVector Y:\n{self._vector_y}"
