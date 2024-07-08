import argparse
import pandas as pd
import os
from pathlib import Path
from amltk.optimization import Metric
from amltk.pipeline import Choice, Sequential, Split
from sklearn.metrics import get_scorer
from sklearn.preprocessing import *
from src.amltk.classifiers.Classifiers import *
from src.amltk.datasets.Datasets import *
from src.amltk.evaluation.Evaluator import get_cv_evaluator
from src.amltk.optimizer.RandomSearch import RandomSearch

from src.amltk.feature_engineering.autofeat.Autofeat import get_autofeat_features
from src.amltk.feature_engineering.AutoGluon.AutoGluon import get_autogluon_features
from src.amltk.feature_engineering.FETCH.FETCH import get_xxx_features
from src.amltk.feature_engineering.H2O.H2O import get_h2o_features
from src.amltk.feature_engineering.OpenFE.OpenFE import get_openFE_features


preprocessing = Split(
    {
        "numerical": Component(SimpleImputer, space={"strategy": ["mean", "median"]}),
        "categorical": [
            Component(
                OrdinalEncoder,
                config={
                    "categories": "auto",
                    "handle_unknown": "use_encoded_value",
                    "unknown_value": -1,
                    "encoded_missing_value": -2,
                },
            ),
            Choice(
                "passthrough",
                Component(
                    OneHotEncoder,
                    space={"max_categories": (2, 20)},
                    config={
                        "categories": "auto",
                        "drop": None,
                        "sparse_output": False,
                        "handle_unknown": "infrequent_if_exist",
                    },
                ),
                name="one_hot",
            ),
        ],
    },
    name="preprocessing",
)

lgbm_classifier = get_lgbm_classifier()
lgbm_classifier_pipeline = Sequential(preprocessing, lgbm_classifier, name="lgbm_classifier_pipeline")

def safe_dataframe(df, working_dir, dataset_name, fold_number, method_name):
    file_string = f"results_{dataset_name}_{method_name}_fold_{fold_number}.parquet"
    results_to = working_dir / file_string
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df)
    print(f"Saving dataframe of results to path: {results_to}")
    df.to_parquet(results_to)


def main():
    parser = argparse.ArgumentParser(description='Run feature engineering methods')
    parser.add_argument('--method', type=str, required=True, help='Feature engineering method to use')
    args = parser.parse_args()

    method = args.method

    rerun = True  # Decide if you want to re-execute the methods on a dataset or use the existing files
    debugging = False  # Decide if you want ot raise trial exceptions
    feat_eng_steps = 2  # Number of feature engineering steps for autofeat
    feat_sel_steps = 5  # Number of feature selection steps for autofeat
    working_dir = Path("src/amltk/results")   # Path if running on Cluster
    # working_dir = Path("results")  # Path for local execution
    random_seed = 42  # Set seed
    folds = 10  # Set number of folds (normal 10, test 1)

    # Choose set of datasets
    all_datasets = [1, 5, 14, 15, 16, 17, 18, 21, 22, 23, 24, 27, 28, 29, 31, 35, 36]  # 17
    small_datasets = [1, 5, 14, 16, 17, 18, 21, 27, 31, 35, 36]
    smallest_datasets = [14, 16, 17, 21, 35]  # n ~ 1000, p ~ 15
    big_datasets = [15, 22, 23, 24, 28, 29]
    test_new_method_datasets = [18]  # [16]

    optimizer_cls = RandomSearch
    pipeline = lgbm_classifier_pipeline

    metric_definition = Metric(
        "roc_auc_ovo",
        minimize=False,
        bounds=(0, 1),
        fn=get_scorer("roc_auc_ovo")
    )

    per_process_memory_limit = None  # (4, "GB")  # NOTE: May have issues on Mac
    per_process_walltime_limit = None  # (60, "s")

    if debugging:
        max_trials = 1  # don't care about quality of the found model
        max_time = 600  # 10 minutes
        n_workers = 20
        # raise an error with traceback, something went wrong
        on_trial_exception = "raise"
        display = True
        wait_for_all_workers_to_finish = False
    else:
        max_trials = 100000  # trade-off between exploration and resource usage
        max_time = 3600  # one hour
        n_workers = 4
        # Just mark the trial as fail and move on to the next one
        on_trial_exception = "continue"
        display = True
        wait_for_all_workers_to_finish = False

    for fold in range(folds):
        print(f"\n\n\n*******************************\n Fold {fold}\n*******************************\n")
        inner_fold_seed = random_seed + fold
        for option in test_new_method_datasets:
            train_x, train_y, test_x, test_y, task_hint, name = get_dataset(option=option)

            if method == "original":
                print("Original Data")
                file_name = f"results_{name}_original_fold_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    evaluator = get_cv_evaluator(train_x, train_y, test_x, test_y, inner_fold_seed, on_trial_exception,
                                                 task_hint)
                    history = pipeline.optimize(
                        target=evaluator.fn,
                        metric=metric_definition,
                        optimizer=optimizer_cls,
                        seed=inner_fold_seed,
                        max_trials=max_trials,
                        timeout=max_time,
                        display=display,
                        wait=wait_for_all_workers_to_finish,
                        n_workers=n_workers,
                        on_trial_exception=on_trial_exception
                    )
                    df = history.df()
                    safe_dataframe(df, working_dir, name, fold, "original")

            elif method == "autofeat":
                print("autofeat Data")
                file_name = f"results_{name}_autofeat_fold_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x_autofeat, test_x_autofeat = get_autofeat_features(train_x, train_y, test_x, task_hint,
                                                                              feat_eng_steps, feat_sel_steps)
                    evaluator = get_cv_evaluator(train_x_autofeat, train_y, test_x_autofeat, test_y, inner_fold_seed,
                                                 on_trial_exception, task_hint)
                    history = pipeline.optimize(
                        target=evaluator.fn,
                        metric=metric_definition,
                        optimizer=optimizer_cls,
                        seed=inner_fold_seed,
                        max_trials=max_trials,
                        timeout=max_time,
                        display=display,
                        wait=wait_for_all_workers_to_finish,
                        n_workers=n_workers,
                        on_trial_exception=on_trial_exception
                    )
                    df = history.df()
                    safe_dataframe(df, working_dir, name, fold, "autofeat")
            """
            elif method == "autogluon":
                print("autogluon Data")
                file_name = f"results_{name}_autogluon_fold_{fold}.parquet"
                file = working_dir / file_name
                if rerun or not os.path.isfile(file):
                    train_x_autogluon, test_x_autogluon = get_autogluon_features(train_x, train_y, test_x)
                    evaluator = get_cv_evaluator(train_x_autogluon, train_y, test_x_autogluon, test_y,
                                                 inner_fold_seed,
                                                 on_trial_exception, task_hint)
                    history = pipeline.optimize(
                        target=evaluator.fn,
                        metric=metric_definition,
                        optimizer=optimizer_cls,
                        seed=inner_fold_seed,
                        max_trials=max_trials,
                        timeout=max_time,
                        display=display,
                        wait=wait_for_all_workers_to_finish,
                        n_workers=n_workers,
                        on_trial_exception=on_trial_exception
                    )
                    df = history.df()
                    safe_dataframe(df, working_dir, name, fold, "autogluon")
            """