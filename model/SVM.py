import numpy as np

class SVC:
    # initiating the hyperparameter
    '''
        learning_rate: The learning speed of the algorithm. This is the step size in updating the weights.
        no_of_iterations: The number of loops to update weights and bias.
        lambda_parameter: Regularization, which helps prevent overfitting by controlling the magnitude of the weights.
        model: Store models (w, b) for each class
    '''

    def __init__(self, learning_rate, no_of_iterations, lambda_parameter):
        self.learning_rate = learning_rate
        self.no_of_iterations = no_of_iterations
        self.lambda_parameter = lambda_parameter
        self.models = []

    # Fitting the dataset to SVC
    def fit(self, X, y):
        # Identify unique labels from y_train
        self.classes = np.unique(y)
        self.models = []

        # Train a binary SVC per class(one-vs-all)
        for class_label in self.classes:
            # Create binary labels for one-vs-all: 1 for current labels, -1 for other labels
            binary_y = np.where(y == class_label, 1, -1)

            # w is the weight vector with a value of 0 for all features.
            # b is the bias, initiates an initial bias value of 0.
            w = np.zeros(X.shape[1])
            b = 0

            # Binary SVC model training for the current class
            for _ in range(self.no_of_iterations):

                # index: This is the index of the current element in the self.X array. It starts at 0
                # x_i: This is the data point that corresponds to that metric 
                for index, x_i in enumerate(X):

                    # Check if the current data template satisfies the SVC constraint
                    condition = binary_y[index] * (np.dot(x_i, w) - b) >= 1

                    if condition:

                        # The SVC does not need to update the weight much
                        # Adjust the weight lightly in the direction of preventing overfitting 
                        dw = 2 * self.lambda_parameter * w
                        db = 0

                    else:

                        # Update weight and update bias
                        dw = 2 * self.lambda_parameter * w - (x_i * binary_y[index])
                        db = binary_y[index]

                    # Use gradient descent to update weights and bias.
                    # The w weights and b bias are adjusted based on the gradient and the learning_rate
                    w -= self.learning_rate * dw
                    b -= self.learning_rate * db

            # save weights and biases for the current class model
            self.models.append((w, b))


    # Predict the label for a given input value
    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes)))
        # Calculate the predicted score for each class
        for i, (w, b) in enumerate(self.models):
            scores[:, i] = np.dot(X, w) - b
        # Returns the label with the highest score
        return self.classes[np.argmax(scores, axis = 1)]


    # Evaluate the accuracy of the model
    def score(self, X, y):
        # Label prediction for input data
        prediction = self.predict(X)
        # Calculate accuracy by comparing predicted labels with actual labels
        accuracy = np.mean(prediction == y)
        return accuracy






