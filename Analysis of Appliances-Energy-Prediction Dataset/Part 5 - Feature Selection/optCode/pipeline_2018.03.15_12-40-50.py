import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:0.25004363652697864
exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_classif, percentile=81),
    GradientBoostingClassifier(learning_rate=0.001, max_depth=7, max_features=0.9500000000000001, min_samples_leaf=3, min_samples_split=14, n_estimators=100, subsample=0.45)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
