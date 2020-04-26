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
       

    def _kernelize_train(self, feature_set_train, kernel_param=[]):
        number_of_features = feature_set_train.shape[0]

        # Other kernels can be added
        # Linear kernel       
        if self.param_kernel_function == "linear":
            ktr = np.dot(feature_set_train, feature_set_train.conjugate().transpose())

        return ktr


    def _kernelize_test(self, feature_set_train,feature_set_test, kernel_param=[] ):
        number_of_features = feature_set_train.shape[0]

        # Other kernels can be added
        # Linear kernel       
        if self.param_kernel_function == "linear":
            kts = np.dot(feature_set_test, feature_set_train.conjugate().transpose())

        return kts

   
    def train(self, X_train, y_train, weighted=False):
        self.X_train = X_train

        number_training_rows = self.X_train.shape[0]
        self.training_patterns = X_train

        # Creating kernels
        kernel_training = self._kernelize_train(self.X_train)

        # 
        classes = np.sort(np.unique(y_train))
        number_of_classes = np.size(classes)
        
        TM = np.zeros((number_training_rows, number_of_classes))

        for i in range(0,number_training_rows):
            for j in range(0,number_of_classes):
                if (j+1) == y_train[i]:
                    TM[i,j] = 1 

        TM = 2*TM - 1

        idenity_matrix = np.identity(number_training_rows)

        if weighted:
            W = np.identity(number_training_rows)
            ClassHistogram = np.histogram(y_train,bins=[1,2,3,4])[0]
            ClassHistogram = np.divide(1,ClassHistogram)
            for i in range(0,number_training_rows):
                W[i,i]=ClassHistogram[y_train[i]-1]
            
            beta = np.linalg.lstsq(((idenity_matrix/self.param_c) + (kernel_training*W)) , TM)
        else:
            beta = np.linalg.lstsq(((idenity_matrix/self.param_c) + kernel_training) , TM)

        self.output_weight = beta[0]
        return beta[0]


    def test(self,testing_features):
        
        kernel_test = self._kernelize_test(self.training_patterns, testing_features,self.param_kernel_params)

        testing_predicted_targets = np.dot(kernel_test.conjugate(), self.output_weight)

        y_argmax = []

        for i in range(0, testing_predicted_targets.shape[0]):
            y_argmax.append(testing_predicted_targets[i].argmax() + 1)

        return y_argmax