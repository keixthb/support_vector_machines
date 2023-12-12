from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def plot_everything(training_data, value_of_c, hilbert_space, risk, emperical_risk):
    my_classifier = svm.SVC(kernel="linear", C=value_of_c).fit(training_data.get_matrix_x(), training_data.get_vector_y())


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))


    ax1.scatter(training_data.get_matrix_x()[:, 0], training_data.get_matrix_x()[:, 1], c=training_data.get_vector_y(), s=30, cmap=plt.cm.Paired)

    DecisionBoundaryDisplay.from_estimator(my_classifier, training_data.get_matrix_x(), plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"], ax=ax1)

    ax1.scatter(my_classifier.support_vectors_[:, 0], my_classifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    ax1.set_title(f'linear classifier using c={value_of_c}')




    ax2.semilogx(hilbert_space, emperical_risk, label = "emperical_risk")
    ax2.semilogx(hilbert_space, risk, label = "risk")
    ax2.semilogx(hilbert_space, risk - emperical_risk, label = "risk - emperical_risk")
    ax2.axvline(x=value_of_c, color='r', linestyle='--', label=f'c={value_of_c}')
    ax2.set_xlabel('values of c')
    ax2.set_ylabel('average error')
    ax2.set_title('risk')
    ax2.legend()



    fig.suptitle('Support Vector Machine')

    plt.savefig("gaussian_support_vector_machine.png")
    plt.show()


    plt.cla()


    return




def plot_support_vector_machine(training_data, value_of_c):
    my_classifier = svm.SVC(kernel="linear", C=value_of_c).fit(training_data.get_matrix_x(), training_data.get_vector_y())

    plt.scatter(training_data.get_matrix_x()[:, 0], training_data.get_matrix_x()[:, 1], c=training_data.get_vector_y(), s=30, cmap=plt.cm.Paired)

    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(my_classifier, training_data.get_matrix_x(), plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"], ax=ax)

    ax.scatter(my_classifier.support_vectors_[:, 0], my_classifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")

    plt.title(f"training_data")
    plt.savefig("support_vector_machine.png")
    plt.show()
    plt.cla()


    '''
    plt.scatter(testing_data.get_matrix_x()[:, 0], testing_data.get_matrix_x()[:, 1], c=testing_data.get_vector_y(), s=30, cmap=plt.cm.Paired)
    ax = plt.gca()

    DecisionBoundaryDisplay.from_estimator(my_classifier, testing_data.get_matrix_x(), plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"], ax=ax)

    ax.scatter(my_classifier.support_vectors_[:, 0], my_classifier.support_vectors_[:, 1], s=100, linewidth=1, facecolors="none", edgecolors="k")


    plt.title(f"testing data")
    '''

    return





def plot_error(hilbert_space, risk, emperical_risk):
    fig, (ax1, ax2) = plt.subplots(1, 2)


    ax2.semilogx(hilbert_space, emperical_risk, label = "emperical_risk")
    ax2.semilogx(hilbert_space, risk, label = "risk")
    ax2.semilogx(hilbert_space, risk - emperical_risk, label = "risk - emperical_risk")
    ax2.legend()

    ax1.semilogx(hilbert_space, emperical_risk, label = "emperical_risk")
    ax1.semilogx(hilbert_space, risk, label = "risk")
    ax1.semilogx(hilbert_space, risk - emperical_risk, label = "risk - emperical_risk")
    ax1.legend()

    fig.suptitle('Support Vector Machine')

    plt.show()



    return
