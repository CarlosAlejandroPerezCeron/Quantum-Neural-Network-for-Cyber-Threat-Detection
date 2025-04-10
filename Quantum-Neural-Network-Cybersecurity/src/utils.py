import pandas as pd
from sklearn.datasets import make_classification

def generate_data():
    X, y = make_classification(n_samples=200, n_features=4, n_informative=3,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    df = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4'])
    df['label'] = y
    return df
