from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
#import../feature_models/oversample.py

#DO NOT DELETE
np.random.seed(42)
class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    model: sci-kit learn model
    test: unlabaled data
    sample_rate = % of unlabeled (test) data to pseudolabel
    features: x
    target: y (label)
    seed: random seeeed
    '''
    def __init__(self, model, labaled_x, labaled_y, unlabaled_x, sample_rate=0.2, seed=42):
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed

        self.labaled_x = labaled_x
        self.labaled_y = labaled_y
        self.unlabaled_x = unlabaled_x
        np.random.seed(self.seed)
        #self.test = test
        #self.features = features
        #self.target = target

    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "labaled_x": self.labaled_x ,
            "labaled_y": self.labaled_y,
            "unlabaled_x": self.unlabaled_x
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        if self.sample_rate > 0.0:
            augemented_train_data, augemented_train_labels = self.__create_augmented_train(X, y)
            self.model.fit(
                augemented_train_data,
                augemented_train_labels
            )
        else:
            self.model.fit(X, y)

        return self

    def __random_sample(self, matrix, num_row):
        number_of_rows = matrix.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_row, replace=False)
        random_rows = matrix[random_indices, :]
        return random_rows


    def __create_augmented_train(self, X, y):

        num_of_samples = int(X.shape[0] * self.sample_rate)

        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        #pseudo_labels = self.model.predict(self.test[self.features])
        pseudo_labels = self.model.predict(self.unlabaled_x)
        pseudo_labels = pseudo_labels.reshape((pseudo_labels.shape[0],1)) #Column of predictions
        augmented_test = np.hstack((self.unlabaled_x,pseudo_labels))

        # Add the pseudo-labels to the test set
        #augmented_test = test.copy(deep=True)

        #augmented_test[self.target] = pseudo_labels

        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set



        #sampled_test = augmented_test.sample(n=num_of_samples)
        sampled_test = self.__random_sample(augmented_test,num_of_samples)
        y = y.reshape((y.shape[0],1))
        temp_train = np.hstack((X,y))
        #augemented_train = pd.concat([sampled_test, temp_train])
        augemented_train = np.vstack((sampled_test,temp_train))
        np.random.shuffle(augemented_train)

        augemented_train_labels  = augemented_train[:,-1]
        augemented_train_data = np.delete(augemented_train, -1, 1)


        return augemented_train_data, augemented_train_labels



    def predict(self, X):
        return self.model.predict(X)

    def get_model_name(self):
        return self.model.__class__.__name__
#model, labaled_x, labaled_y, unlabaled_x, sample_rate=0.2, seed=42
