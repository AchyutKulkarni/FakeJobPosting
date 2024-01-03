from my_evaluation import my_evaluation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class my_model():

    def __init__(self):
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self):
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=True)
        knn_classifier = KNeighborsClassifier()

        pipeline = Pipeline([
            ('tfidf', tfidf_vectorizer),
            ('knn', knn_classifier)
        ])

        return pipeline

    def obj_func(self, predictions, actuals, pred_proba=None):
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    def fit(self, X, y, param_grid=None, cv=5):
        if param_grid is None:
            param_grid = {
                'tfidf__max_features': [5000, 10000],
                'knn__n_neighbors': [3, 5, 7],
                'knn__weights': ['distance', 'uniform']
            }

        grid_search = GridSearchCV(self.pipeline, param_grid=param_grid, cv=cv, scoring='f1')
        grid_search.fit(X["description"], y)

        self.pipeline = grid_search.best_estimator_

    def predict(self, X, threshold=0.5):
        if not hasattr(self, 'pipeline'):
            raise ValueError("Model not fitted. Call fit() before predict()")

        predicted_probs = self.pipeline.predict_proba(X["description"])
        predictions = (predicted_probs[:, 1] > threshold).astype(int)

        return predictions
