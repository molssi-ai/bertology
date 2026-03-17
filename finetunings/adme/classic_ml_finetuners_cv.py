###############################################################################
# Author: Mohammad Mostafanejad                                               #
# Date: December 2025                                                         #
# Description:                                                                #
# This script performs a nested cross-validation hyperparameter search on     #
# Biogen's public ML data. The featurized csv data files must be generated    #
# by the "adme_ml_public.py" script.                                          #
###############################################################################

# import the necessary libraries
import os
import csv
import json
from tqdm import tqdm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import pearsonr
from datasets import load_from_disk, DatasetDict, Dataset
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, GridSearchCV, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def main(ds_name: str):
    # set the main variables
    input_data_path = os.path.join(os.getcwd(), "data", "adme_ml_public")
    output_dir = os.path.join(os.getcwd(), "data")
    seed = 1234
    n_jobs_cv = 2
    folds_to_huggingface = True  # save the fold splits as huggingface datasets on disk
    store_cv_indices = True  # save the train/validation indices of each fold to json

    # read the dataset from disk
    ds = load_from_disk(input_data_path)[ds_name]

    # hyperparameter search space
    params = {
        "LASSO": {"alpha": 0.1},
        "search_LASSO": {"LASSO__alpha": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5]},
        "RF": {
            "n_estimators": 500,
            "oob_score": True,
            "n_jobs": n_jobs_cv,
            "random_state": seed,
        },
        "search_RF": {
            "n_estimators": [100, 250, 500, 750, 1000],
            "max_features": ["sqrt", 0.33, 0.67, None],
            "max_depth": [15, 25, 40, None],
        },
        "SVM": {"gamma": "scale"},
        "search_SVM": {
            "SVM__C": [0.1, 1, 5, 10, 20, 50],
            "SVM__epsilon": [1e-2, 1e-1, 0.3, 0.5],
            "SVM__gamma": ["scale", "auto"],
        },
        "XGB": {
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": n_jobs_cv,
            "random_state": seed,
            "device": "cuda",
            "nthread": 4,
        },
        # "search_XGB": {
        #     "n_estimators": [100, 250, 500, 750, 1000],
        #     "max_depth": [3, 4, 5, 6, 7],
        #     "min_child_weight": [1, 2, 3],
        #     "gamma": [0, 0.05, 0.1],
        #     "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        #     "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        #     "reg_alpha": [0, 0.1, 0.2, 0.3, 0.4],
        #     "reg_lambda": [1, 1.1, 1.2, 1.3, 1.4],
        # },
        "search_XGB": {
            "round_1": {"n_estimators": [100, 250, 500, 750, 1000]},
            "round_2": {"max_depth": [3, 4, 5, 6, 7], "min_child_weight": [1, 2, 3]},
            "round_3": {"gamma": [0, 0.05, 0.1]},
            "round_4": {
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            },
            "round_5": {
                "reg_alpha": [0, 0.1, 0.2, 0.3, 0.4],
                "reg_lambda": [1, 1.1, 1.2, 1.3, 1.4],
            },
        },
        "LGBM": {
            "n_estimators": 500,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "subsample_freq": 1,
            "n_jobs": n_jobs_cv,
            "random_state": seed,
            "device": "gpu",
        },
        # "search_LGBM": {
        #     "n_estimators": [100, 250, 500, 750, 1000],
        #     "num_leaves": [15, 31, 45, 60, 75],
        #     "min_child_samples": [10, 20, 30, 40],
        #     "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        #     "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        #     "subsample_freq": [0, 1, 3, 5],
        #     "reg_alpha": [0, 0.2, 0.5, 0.8],
        #     "reg_lambda": [0, 0.2, 0.5, 0.8],
        # },
        "search_LGBM": {
            "round_1": {
                "n_estimators": [100, 250, 500, 750, 1000],
            },
            "round_2": {
                "num_leaves": [15, 31, 45, 60, 75],
                "min_child_samples": [10, 20, 30, 40],
            },
            "round_3": {
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "subsample_freq": [0, 1, 3, 5],
            },
            "round_4": {
                "reg_alpha": [0, 0.2, 0.5, 0.8],
                "reg_lambda": [0, 0.2, 0.5, 0.8],
            },
        },
    }

    # model dicts
    models = {
        "LASSO": Pipeline(
            [("scaler", RobustScaler()), ("LASSO", Lasso(**params["LASSO"]))]
        ),
        "RF": RandomForestRegressor(**params["RF"]),
        "SVM": Pipeline([("scaler", RobustScaler()), ("SVM", SVR(**params["SVM"]))]),
        "XGB": XGBRegressor(**params["XGB"]),
        "LGBM": LGBMRegressor(**params["LGBM"]),
    }

    # define scoring metrics
    scores = {
        "pearson_r": make_scorer(
            lambda y_true, y_pred: pearsonr(y_true, y_pred)[0], greater_is_better=True
        ),
        "mae": make_scorer(
            lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
            greater_is_better=False,
        ),
        "r2": make_scorer(
            lambda y_true, y_pred: r2_score(y_true, y_pred), greater_is_better=True
        ),
        "rmse": make_scorer(
            lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
            greater_is_better=False,
        ),
    }

    # prepare training data
    split_ds = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)

    X_train = (
        split_ds["train"]
        .to_pandas()
        .drop(
            [
                "activity",
                "rdkit_SMILES",
                "molecule_ID",
                "pubchem_SMILES",
                "pubchem_cid",
            ],
            axis=1,
        )
    )
    Y_train = split_ds["train"].to_pandas()["activity"]
    # # prepare test data (hold-out set)
    X_test = (
        split_ds["test"]
        .to_pandas()
        .drop(
            [
                "activity",
                "rdkit_SMILES",
                "molecule_ID",
                "pubchem_SMILES",
                "pubchem_cid",
            ],
            axis=1,
        )
    )
    Y_test = split_ds["test"].to_pandas()["activity"]

    # create 3 folds inner-/outer-KFold objects for nested cross-validation
    # see https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html#sphx-glr-auto-examples-model-selection-plot-nested-cross-validation-iris-py
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)

    # perform nested cross-validation
    # perform cross-validation
    with open(
        os.path.join(output_dir, f"cv_scores_{ds_name}.csv"), mode="w", newline=""
    ) as csv_file:
        # create the csv writer object
        csv_writer = csv.writer(csv_file)
        # write the score_names on the columns
        csv_writer.writerow(
            ["model name"]
            + [f"mean_train_{score_name}" for score_name in scores.keys()]
            + [f"mean_valid_{score_name}" for score_name in scores.keys()]
            + [f"test_{score_name}" for score_name in scores.keys()]
            + [f"std_train_{score_name}" for score_name in scores.keys()]
            + [f"std_valid_{score_name}" for score_name in scores.keys()]
        )
        # loop over models
        for model_name, model in tqdm(
            models.items(), desc=f"Processing {ds_name}", total=len(models)
        ):
            if model_name in ["XGB", "LGBM"]:
                # go through hyperparameter search rounds
                for idx, param_round in params[f"search_{model_name}"].items():
                    # loop over scoring metrics using X_tr and Y_tr
                    grid_search_cv = GridSearchCV(
                        estimator=model,
                        param_grid=param_round,
                        scoring=scores,
                        n_jobs=n_jobs_cv,
                        cv=kf,
                        refit="r2",
                        return_train_score=True,
                    )
                    # perform grid search cv
                    gs_cv = grid_search_cv.fit(X_train, Y_train)
                    cv_results = gs_cv.cv_results_
                    # update model with best params from this round
                    model.set_params(**gs_cv.best_params_)
            else:  # if not XGB or LGBM, proceed as usual
                # loop over scoring metrics using X_tr and Y_tr
                grid_search_cv = GridSearchCV(
                    estimator=model,
                    param_grid=params[f"search_{model_name}"],
                    scoring=scores,
                    n_jobs=n_jobs_cv,
                    cv=kf,
                    refit="r2",
                    return_train_score=True,
                )
                # perform grid search cv
                gs_cv = grid_search_cv.fit(X_train, Y_train)
                cv_results = gs_cv.cv_results_

            # store the best model parameters
            with open(
                os.path.join(output_dir, f"best_params_{ds_name}_{model_name}.json"),
                "w",
            ) as json_file:
                json.dump(gs_cv.best_params_, json_file, indent=4)

            # save the fold splits as huggingface datasets on disk
            if folds_to_huggingface:
                # use fold train and validation indices to create the huggingface dataset splits
                folded_ds = DatasetDict()
                for fold_idx, (train_indices, valid_indices) in enumerate(
                    gs_cv.cv.split(X_train, Y_train)
                ):
                    folded_ds[f"fold_{fold_idx}_train"] = (
                        split_ds["train"]
                        .select_columns(
                            [
                                "activity",
                                "rdkit_SMILES",
                                "molecule_ID",
                                "pubchem_SMILES",
                                "pubchem_cid",
                            ]
                        )
                        .select(train_indices)
                    )
                    folded_ds[f"fold_{fold_idx}_valid"] = (
                        split_ds["train"]
                        .select_columns(
                            [
                                "activity",
                                "rdkit_SMILES",
                                "molecule_ID",
                                "pubchem_SMILES",
                                "pubchem_cid",
                            ]
                        )
                        .select(valid_indices)
                    )

                # store the test set
                folded_ds["test"] = split_ds["test"].select_columns(
                    [
                        "activity",
                        "rdkit_SMILES",
                        "molecule_ID",
                        "pubchem_SMILES",
                        "pubchem_cid",
                    ]
                )

                # save the folded dataset to disk
                folded_ds.save_to_disk(
                    os.path.join(
                        output_dir, f"adme_folded_hf_ds_{ds_name}_{model_name}"
                    )
                )

            # store the fold indices to json
            if store_cv_indices:
                with open(
                    os.path.join(output_dir, f"cv_indices_{ds_name}_{model_name}.json"),
                    "w",
                ) as json_file:
                    indices_dict = {}
                    for fold_idx, (train_indices, val_indices) in enumerate(
                        gs_cv.cv.split(X_train, Y_train)
                    ):
                        indices_dict[f"fold_{fold_idx}"] = {
                            "train_indices": train_indices.tolist(),
                            "val_indices": val_indices.tolist(),
                        }
                    json.dump(indices_dict, json_file, indent=4)

            # get the best estimator and make predictions on the hold-out test set
            Y_pred = gs_cv.best_estimator_.predict(X_test)

            # write average scores to csv
            # average training scores across folds
            # print(cv_results)
            train_avg_scores = [
                cv_results[f"mean_train_{score_name}"][gs_cv.best_index_]
                for score_name in scores.keys()
            ]
            valid_avg_scores = [
                cv_results[f"mean_test_{score_name}"][gs_cv.best_index_]
                for score_name in scores.keys()
            ]
            # std dev training scores across folds
            train_std_scores = [
                cv_results[f"std_train_{score_name}"][gs_cv.best_index_]
                for score_name in scores.keys()
            ]
            valid_std_scores = [
                cv_results[f"std_test_{score_name}"][gs_cv.best_index_]
                for score_name in scores.keys()
            ]

            # calculate test set scores
            test_scores = {
                "pearson_r": pearsonr(Y_test, Y_pred)[0],
                "mae": mean_absolute_error(Y_test, Y_pred),
                "r2": r2_score(Y_test, Y_pred),
                "rmse": root_mean_squared_error(Y_test, Y_pred),
            }

            # write test set scores to csv
            csv_writer.writerow(
                [model_name]
                + train_avg_scores
                + valid_avg_scores
                + list(test_scores.values())
                + train_std_scores
                + valid_std_scores
            )


if __name__ == "__main__":
    # set the dataset names
    ds_names = ["HLM", "hPPB", "MDR1_ER", "RLM", "rPPB", "SOL"]
    for ds_name in ds_names:
        main(ds_name=ds_name)
