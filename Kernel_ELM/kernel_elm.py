import numpy as np
import pandas as pd
import scipy


class Extreme_Learning_Machine():
    def __init__(self, kernel="linear", C=1, kernel_params = [-15]):
        super(self.__class__, self).__init__()
        
        self.regressor_name = "kernel_elm"

        self.available_kernel_functions = ["linear"]   

        self.output_weight = []
        self.training_patterns = []
        
        self.param_kernel_function = kernel
        self.param_c = C
        self.param_kernel_params = kernel_params
       

    def _kernel_matrix(self, training_patterns, kernel_type, kernel_param, test_patterns="training"):
        number_training_patterns = training_patterns.shape[0]

        # Other kernels can be added
        # Linear kernel       
        if kernel_type == "linear":
            if test_patterns is "training":
                ktr = np.dot(training_patterns, training_patterns.conjugate().transpose())
            elif test_patterns is "testing":
                ktr = np.dot(training_patterns, test_patterns.conjugate().transpose())

        return ktr


   
    def fit(self, training_patterns, training_expected_targets, params=[]):
        self.training_patterns = training_patterns
        number_training_patterns = self.training_patterns.shape[0]
        
        # Traning phase
        omega_train = self._kernel_matrix(self.training_patterns, self.param_kernel_function, self.param_kernel_params)

        self.output_weight = np.linalg.solve((omega_train + np.eye(number_training_patterns)/(2 ** self.param_c)), training_expected_targets).reshape(-1,1)

        training_predicted_targets = np.dot(omega_train, self.output_weight)

        return training_predicted_targets


    def test(self,testing_features, testing_labels, predicting=False):
        
        omega_test = self._kernel_matrix(self.training_patterns, self.param_kernel_function, self.param_kernel_params, testing_patterns)

        testing_predicted_targets = np.dot(omega_test.conjugate().transpose(), self.output_weight)

        return testing_predicted_targets    



    #def predict(self, horizon=1):
    #    return self._ml_predict(horizon)





