import warnings
import os

import pandas as pd
from src.autogluon.method.tabular.src.autogluon.tabular import (TabularDataset, TabularPredictor)
from sklearn.exceptions import UndefinedMetricWarning

from src.datasets.Datasets import preprocess_data, preprocess_target
from src.datasets.Splits import get_splits

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UndefinedMetricWarning)

dataset_files = sorted(os.listdir("../datasets/feature_engineered_datasets"))
old_method = None
old_dataset = None
eval_df = pd.DataFrame()
leaderboard = pd.DataFrame()
for dataset_file in dataset_files:
    if dataset_file.endswith(".parquet") and not dataset_file.__contains__("exec_times"):
        core_name = dataset_file[:-len('.parquet')]
        parts = core_name.split('_')
        task_hint = parts[0]
        method = parts[-2]
        fold = parts[-1]
        parts.remove(task_hint)
        parts.remove(method)
        parts.remove(fold)
        dataset = '_'.join(parts)

        print(
            f"\n****************************************\n {dataset} - {method} - {fold} \n****************************************")
        try:
            execution_times = pd.read_parquet(
                f"../datasets/feature_engineered_datasets/exec_times/exec_times_{dataset}_{method}_{fold}.parquet")
            result = execution_times[(execution_times['Dataset'] == dataset) & (execution_times['Method'] == method)]
            exec_time = result['Time'].values[0]
        except FileNotFoundError:
            exec_time = 0  # no FE method executed on dataset -> raw dataset

        num_cpus = 8
        memory_limit = 32
        time_limit = 14400 - exec_time  # 4h in seconds - time needed for feature engineering
        print(f"Time limit: {time_limit}")

        leaderboard = pd.DataFrame()
        eval_dict = None
        if time_limit >= 0:
            try:
                data = pd.read_parquet(
                    f'../datasets/feature_engineered_datasets/regression_{dataset}_{method}_{fold}.parquet')
                task_hint = 'regression'
            except:
                try:
                    data = pd.read_parquet(
                        f'../datasets/feature_engineered_datasets/binary-classification_{dataset}_{method}_{fold}.parquet')
                    task_hint = 'binary'
                except:
                    data = pd.read_parquet(
                        f'../datasets/feature_engineered_datasets/multi-classification_{dataset}_{method}_{fold}.parquet')
                    task_hint = 'multiclass'
            label = data.columns[-1]

            X = data.drop(label, axis=1)
            y = data[label]

            train_x, train_y, test_x, test_y = get_splits(X, y, 42)

            train_x, test_x = preprocess_data(train_x, test_x)
            train_y = preprocess_target(train_y)
            test_y = preprocess_target(test_y)

            train_data = pd.concat([train_x, train_y], axis=1)
            test_data = pd.concat([test_x, test_y], axis=1)
            train_data = TabularDataset(train_data)
            test_data = TabularDataset(test_data)
            path = "logs/autogluon_" + dataset + "_" + method + "_" + fold + ".parquet"

            if task_hint == 'regression':
                predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint,
                                             eval_metric="root_mean_squared_error").fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                    memory_limit=memory_limit)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            elif task_hint == 'binary':
                predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint,
                                             eval_metric="roc_auc").fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                    memory_limit=memory_limit)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            elif task_hint == 'multiclass':
                predictor = TabularPredictor(label=label, verbosity=0, problem_type=task_hint,
                                             eval_metric="log_loss").fit(
                    train_data=train_data, time_limit=time_limit, num_cpus=num_cpus, presets="best_quality",
                    memory_limit=memory_limit)
                eval_dict = predictor.evaluate(test_data)
                leaderboard = predictor.leaderboard(test_data)
            leaderboard.to_parquet(f"../autogluon/results/leaderboard/leaderboard_{dataset}_{method}_{fold}.parquet")
        if method == old_method and dataset == old_dataset:
            eval_df = eval_df._append(pd.DataFrame(eval_dict, index=[0]))
            eval_df.to_parquet(f"../autogluon/results/evaldict/evaldict_{dataset}_{method}.parquet")
        else:
            if old_method is not None and old_dataset is not None:
                eval_df.to_parquet(f"../autogluon/results/evaldict/evaldict_{old_dataset}_{old_method}.parquet")
            eval_df = pd.DataFrame(eval_dict, index=[0])
            eval_df.to_parquet(f"../autogluon/results/evaldict/evaldict_{dataset}_{method}.parquet")
        old_method = method
        old_dataset = dataset
