from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier

def get_ml_models():
    """
    Define machine learning models.
    Returns a dictionary mapping model names to their instances.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear'),
        'Random Forest': RFC(random_state=42),
        'MLP': MLPClassifier(max_iter=1000, random_state=42)
    }
