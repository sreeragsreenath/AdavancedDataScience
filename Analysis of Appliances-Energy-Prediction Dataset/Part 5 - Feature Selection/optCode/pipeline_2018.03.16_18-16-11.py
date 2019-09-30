import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'].values, random_state=42)

# Score on the training set was:-5833.122261872786
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.5, loss="ls", max_depth=9, max_features=0.2, min_samples_leaf=10, min_samples_split=15, n_estimators=300, subsample=0.5)),
    GradientBoostingRegressor(alpha=0.8, learning_rate=0.5, loss="ls", max_depth=4, max_features=0.35000000000000003, min_samples_leaf=10, min_samples_split=8, n_estimators=300, subsample=0.8500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
