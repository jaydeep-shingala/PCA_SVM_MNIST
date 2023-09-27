import numpy as np
from tqdm import tqdm


class PCA:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.eig_val = None
        self.eig_vec = None
        self.eig_pairs = None
    
    def fit(self, X) -> None:
        # fit the PCA model
        meanofdata = np.mean(X, axis= 0)
        mean_centered_data = X - meanofdata
        covmat = np.cov(mean_centered_data.T)
        eig_val_unsorted, eig_vec_unsorted = np.linalg.eig(covmat)
        
        eig_pairs = [(eig_val_unsorted[i], eig_vec_unsorted[:, i]) for i in range(len(eig_val_unsorted))]
        eig_pairs.sort(reverse=True, key=lambda x: x[0])
        self.eig_pairs = eig_pairs
        
        # raise NotImplementedError
    
    def transform(self, X) -> np.ndarray:
        # transform the data
        k = self.n_components
        top_eig_pairs = self.eig_pairs[:k]
        top_eig_vecs = [pair[1] for pair in top_eig_pairs]
        transformed_X = np.dot(X, np.array(top_eig_vecs).T)
        return transformed_X
        
        # raise NotImplementedError

    def fit_transform(self, X) -> np.ndarray:
        # fit the model and transform the data
        self.fit(X)
        return self.transform(X)


class SupportVectorModel:
    def __init__(self) -> None:
        pass
    
    def _initialize(self, X) -> None:
        # initialize the parameters
        self.w = np.zeros(X.shape[1])
        self.b = 0
        pass

    def fit(
            self, X, y, 
            learning_rate: float,
            num_iters: int,
            C: float = 1.0,
    ) -> None:
        self._initialize(X)
        # fit the SVM model using stochastic gradient descent
        for i in tqdm(range(1, num_iters + 1)):
            # sample a random training example
            random_index = np.random.choice(X.shape[0])
            random_sample = X[random_index]
            sample_label = y[random_index]

            t = sample_label * (np.dot(self.w, random_sample.T) + self.b)

            if t > 1:
                gradw = 0
                gradb = 0
            else:
                gradw = C * sample_label * random_sample 
                gradb = C * sample_label

            self.w = self.w - learning_rate * (self.w - gradw)
            self.b = self.b + learning_rate * gradb       
            # raise NotImplementedError
    
    def predict(self, X) -> np.ndarray:
        # make predictions for the given data
        raise NotImplementedError

    def accuracy_score(self, X, y) -> float:
        # compute the accuracy of the model (for debugging purposes)
        return np.mean(self.predict(X) == y)


class MultiClassSVM:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.f1score = np.zeros(self.num_classes)
        self.precision = np.zeros(self.num_classes)
        self.recall = np.zeros(self.num_classes)
        self.models = []
        for i in range(self.num_classes):
            self.models.append(SupportVectorModel())
    
    def fit(self, X, y, **kwargs) -> None:
        # first preprocess the data to make it suitable for the 1-vs-rest SVM model
        # then train the 10 SVM models using the preprocessed data for each class
        if 'learning_rate' in kwargs:
            learning_rate = kwargs['learning_rate']
        if 'num_iters' in kwargs:
            num_itr = kwargs['num_iters']
        for i in range(0,self.num_classes):
            classnum = i
            new_Labels = np.where(y == classnum, 1, -1)
            self.models[classnum].fit(X, new_Labels, learning_rate, num_itr)
        # raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        # pass the data through all the 10 SVM models and return the class with the highest score
        score_list = []
        indexans = 0
        for point in range(0, X.shape[0]):
            maxtillnow = float('-inf')
            for classlabel in range(0, self.num_classes):
                predicted_classprob = np.dot(X[point], (self.models[classlabel].w).T) + self.models[classlabel].b
                predicted_classprob = predicted_classprob
                if( maxtillnow < predicted_classprob):
                    maxtillnow = predicted_classprob
                    indexans = classlabel
            score_list.append(indexans)
        return np.array(score_list)
        # raise NotImplementedError

    def compute_confusion_matrix(self, i, y):
        
    # Calculate true positives, false positives, and false negatives for the current class
        tp = np.sum((y == i) & (self.predicted_y== i))
        fp = np.sum((y != i) & (self.predicted_y == i))
        fn = np.sum((y == i) & (self.predicted_y != i))
        return tp, fp, fn

    def accuracy_score(self, X, y) -> float:
        self.predicted_y = self.predict(X)
        return np.mean(self.predicted_y == y)
    
    def precision_score(self, X, y) -> float:
        for i in range(self.num_classes):
            # Calculate precision, recall, and F1 score for the current class
            tp, fp, fn = self.compute_confusion_matrix(i, y)
            self.precision[i] = tp / (tp + fp)
        return np.mean(self.precision)
        # raise NotImplementedError
    
    def recall_score(self, X, y) -> float:
        for i in range(self.num_classes):
            # Calculate precision, recall, and F1 score for the current class
            tp, fp, fn = self.compute_confusion_matrix(i, y)
            self.recall[i] = tp / (tp + fn)
        return np.mean(self.recall)
        # raise NotImplementedError
    
    def f1_score(self, X, y) -> float:
        for i in range(self.num_classes):
            self.f1score[i] = 2 * (self.precision[i] * self.recall[i]) / (self.precision[i] + self.recall[i])
        return np.mean(self.f1score)
        # raise NotImplementedError
