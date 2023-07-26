

# stdlib
from synthcity.benchmark import Benchmarks
from synthcity.plugins.core.dataloader import GenericDataLoader
from synthcity.plugins import Plugins
import synthcity.logger as log
from sklearn.datasets import load_iris
import sys
import warnings

# two lines from benchmark-> __init__.py
from pathlib import Path
from synthcity.metrics import Metrics
warnings.filterwarnings("ignore")

# third party

# synthcity absolute

X, y = load_iris(return_X_y=True, as_frame=True)
X["target"] = y

loader = GenericDataLoader(X, target_column="target", sensitive_columns=[])

loader.dataframe()


# synthcity absolute

score = Benchmarks.evaluate(
    [("uniform_sampler", "uniform_sampler", {})],
    loader,
    # X_test = loader,
    synthetic_size=len(X),
    repeats=3,
    metrics={
        # 'sanity': ['data_mismatch', 'common_rows_proportion'],
        'stats': ['jensenshannon_dist', 'alpha_precision'],
    }
)
print(score)

X_syn = loader.test()
X_ref_syn = X_syn
metrics = {
    # 'sanity': ['data_mismatch', 'common_rows_proportion'],
    'stats': ['jensenshannon_dist', 'alpha_precision'],
}
task_type = "classification"
use_metric_cache = True
X_augmented = None
workspace = Path("workspace")
evaluation = Metrics.evaluate(
    loader.test(),
    X_syn,
    loader.train(),
    X_ref_syn,
    X_augmented,
    metrics=metrics,
    task_type=task_type,
    workspace=workspace,
    use_cache=use_metric_cache,
)

a = 1
