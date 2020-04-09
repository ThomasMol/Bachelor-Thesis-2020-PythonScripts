import numpy as np

class K_elm_Tools(object):

    def _ml_train(self, training_patterns, training_expected_targets, params):
                training_predicted_targets = self._local_train(training_patterns,training_expected_targets,params)

                # Training erros
                #training_errors = Error(training_expected_targets,training_predicted_targets,regressor_name=self.regressor_name)

                # Save last pattern for posterior predictions
                #self.last_training_pattern = training_matrix[-1, :]
                self.has_trained = True

                # Should return training_erros
                return None
    
    def _ml_test(self, testing_patterns, testing_expected_targets, predicting=False):
       
        testing_expected_targets = testing_expected_targets.reshape(-1, 1)

        testing_predicted_targets = self._local_test(testing_patterns,testing_expected_targets,predicting)

        #testing_errors = Error(testing_expected_targets, testing_predicted_targets, regressor_name=self.regressor_name)

        return None #testing_errors

    def _ml_predict(self, horizon=1):
        # Create first new pattern
        new_pattern = np.hstack([self.last_training_pattern[2:],
                                 self.last_training_pattern[0]])

        # Create a fake target (1)
        new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)

        predicted_targets = np.zeros((horizon, 1))

        for t_counter in range(horizon):
            te_errors = self.test(new_pattern, predicting=True)

            predicted_value = te_errors.predicted_targets
            predicted_targets[t_counter] = predicted_value

            # Create a new pattern including the actual predicted value
            new_pattern = np.hstack([new_pattern[0, 2:],
                                     np.squeeze(predicted_value)])

            # Create a fake target
            new_pattern = np.insert(new_pattern, 0, 1).reshape(1, -1)
            
        return predicted_targets
