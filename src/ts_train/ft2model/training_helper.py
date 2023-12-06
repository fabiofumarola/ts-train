from __future__ import annotations
import pickle
from pathlib import Path

from typing import Optional, Union, Literal, Tuple, Any

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.ml import Transformer, Estimator
from pyspark.ml.feature import (
    StringIndexerModel,
    OneHotEncoderModel,
)
from xgboost.spark import (  # type: ignore
    SparkXGBClassifierModel,
    SparkXGBRegressorModel,
)
from pydantic import BaseModel, ConfigDict
from pydantic.types import StrictStr

from ts_train.ft2model.core import (
    Params,
    TunableParams,
    tune_parameters,
    tune_parameters_cv,
    get_estimator,
    get_evaluator,
    fit,
    predict,
    score,
    label_encode,
    one_hot_encode,
    vector_assemble,
    get_features_importance,
)


class TrainingHelper(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """TrainingHelper is an helper step to train a classification or regression model
    in a distributed and highly optimized way.

    Attributes:
        type (Literal["classification", "regression"]): Type of training model to be
            used. You have to chose between "classification" and "regression".
        features_cols_name (list[StrictStr]): List of columns names of features in the
            DataFrame to be used to fit the model.
        label_col_name (StrictStr): column name of target/label in the DataFrame to be
            used to fit the model for the classification or regression task.
        params (Params, optional): Parameters for the SparkXGBClassifier or
            SparkXGBRegressor model. XGBoost documentation here:
            https://xgboost.readthedocs.io/en/stable/python/python_api.html
        ordered_categ_features_cols_name (list[StrictStr], optional): list of columns
            names which should be a subset of features_cols_name. These are the
            categorical columns with values that should be mainteined an order with.
        unordered_categ_features_cols_name (list[StrictStr], optional): list of columns
            names which should be a subset of features_cols_name. These are the
            categorical columns with values that should not be mainteined an order with.
        num_workers (int): How many XGBoost workers to be used to train. Each XGBoost 
            worker corresponds to one spark task. It can be also provided in the params
            dictionary. This parameter overwrites the params dictionary.

    Raises:
        ValueError: if ordered_categ_features_cols_name is not a subset of
            features_cols_name.
        ValueError: if unordered_categ_features_cols_name is not a subset of
            features_cols_name.
    """

    type: Literal["classification", "regression"]
    features_cols_name: list[StrictStr]
    label_col_name: StrictStr
    ordered_categ_features_cols_name: Optional[list[StrictStr]] = None
    unordered_categ_features_cols_name: Optional[list[StrictStr]] = None
    num_workers: int = 1

    _transformer: Optional[Transformer] = None
    _estimator: Optional[Estimator] = None
    _string_indexer_model: Optional[StringIndexerModel] = None
    _one_hot_encoder_model: Optional[OneHotEncoderModel] = None
    _cols_encoding: Optional[dict[str, list[Union[str, int, float]]]] = None
    _encoded_features_cols_name: Optional[list[str]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.ordered_categ_features_cols_name is not None and not all(
            item in self.features_cols_name
            for item in self.ordered_categ_features_cols_name
        ):
            raise ValueError(
                "Categorical features (ordered_categ_features_cols_name) should be also"
                " included in features_cols_name"
            )
        if self.unordered_categ_features_cols_name is not None and not all(
            item in self.features_cols_name
            for item in self.unordered_categ_features_cols_name
        ):
            raise ValueError(
                "Categorical features unordered_categ_features_cols_name) should be"
                " also included in features_cols_name"
            )

    def fit(
        self, df: DataFrame, objective=None, params: Optional[Params] = None
    ) -> Transformer:
        """Given a DataFrame it fits the model on the features provided in the
        DataFrame.

        Args:
            df (DataFrame): DataFrame containing features and target.
            params (Params, optional): parameters to be passed to the regressor or the
                classifier.
            objective (str, optional): Metric to be used to optimize the model and train
                it. It is usable only with regression: reg:squarederror,
                reg:squaredlogerror, reg:logistic, reg:pseudohubererror,
                reg:absoluteerror, reg:quantileerror. Others could be found here:
                https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
                XGBoost for Spark does not allow to modify objective for classification.
                For regression default is reg:squarederror.

        Raises:
            ValueError: if params contains list of values instead of a strict value.

        Returns:
            Transformer: fitted model.
        """
        if objective is None and self.type == "regression":
            objective = "reg:squarederror"

        df = self._preprocess_dataframe(df)

        if not self._are_params_tunable(params):
            self._estimator = get_estimator(
                type=self.type,
                label_col_name=self.label_col_name,
                objective=objective,
                num_workers=self.num_workers,
                params=params,
            )
            self._transformer = fit(
                df=df,
                estimator=self._estimator,
            )
        else:
            raise ValueError(
                "params parameters should include strict values, not list."
            )

        return self._transformer

    def tune(
        self,
        train_df: DataFrame,
        val_df: DataFrame,
        params: TunableParams,
        objective: Optional[str] = None,
        metric: Optional[str] = None,
        mode: Optional[str] = None,
    ):
        """Given a DataFrame it tunes given params and fits the model on the features
        provided in the train DataFrame and evaluates different parameters with the
        validation DataFrame.

        Args:
            train_df (DataFrame): DataFrame containing features and target. Used to
                fit the models.
            val_df (DataFrame): DataFrame used to evaluate different models and return
                the best one.
            params (TunableParams): dictionary with keys as regressor or classifier
                parameters and values as strict values or list of strict values. Values
                of lists will be tuned.
            objective (str, optional): Metric to be used to optimize the model and train
                it. Available options:
                - Classification: multi:softmax, multi:softprob
                - Binary classification: binary:logistic, binary:logitraw, binary:hinge
                - Regression: reg:squarederror, reg:squaredlogerror, reg:logistic,
                    reg:pseudohubererror, reg:absoluteerror, reg:quantileerror
                Others could be found here: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
                Defaults to "multi:softmax" for classification and "reg:squarederror"
                for regression.
            metric (str, optional): Metric to be used to evaluate the models trained
                with different values of the parameters provided. Available options:
                - Classification: f1, accuracy, weightedPrecision, weightedRecall,
                    weightedTruePositiveRate, weightedFalsePositiveRate,
                    weightedFMeasure, truePositiveRateByLabel,
                    falsePositiveRateByLabel, precisionByLabel, recallByLabel,
                    fMeasureByLabel, logLoss, hammingLoss.
                - Binary classification: areaUnderROC, areaUnderPR, and those of the
                    normal classification.
                - Regression: rmse, mse, r2, mae, var
                Defaults to "f1" for classification and "rmse" for regression.
            mode (str): "min" or "max". If you have accuracy you would like to maximize
                the metric, so you would use "max". If metrix is "f1" it maximses it,
                otherwise it minimize the metric.Default "min".

        Returns:
            Params: parameters used to fit the best performing model considering the
                chosen metric.
            Transformer: fitted model with best performance considering the chosen
                metric.
        """
        if objective is None and self.type == "regression":
            objective = "reg:squarederror"
        if metric is None:
            metric = "f1" if self.type == "classification" else "rmse"
        if mode is None:
            if metric == "f1":
                mode = "max"
            else:
                mode = "min"

        train_df = self._preprocess_dataframe(train_df)
        val_df = self._preprocess_dataframe(val_df)

        if self._are_params_tunable(params):
            best_params, self._transformer, self._estimator = tune_parameters(
                train_df=train_df,
                val_df=val_df,
                params=params,
                type=self.type,
                label_col_name=self.label_col_name,
                objective=objective,
                mode=mode,
                num_workers=self.num_workers,
                evaluator=get_evaluator(
                    metric=metric, label_col_name=self.label_col_name
                ),
            )
        else:
            raise ValueError(
                "params parameters should include at least one list of possible values."
            )

        return best_params, self._transformer

    def tune_cv(
        self,
        df: DataFrame,
        params: TunableParams,
        objective=None,
        metric=None,
        cv_folds: int = 5,
    ) -> Tuple[Params, Transformer]:
        """Given a DataFrame it tunes given params and fits the model on the features
        provided in the DataFrame using cross validation.

        Args:
            df (DataFrame): DataFrame containing features and target. It is internally
                split using cross validation.
            params (TunableParams): dictionary with keys as regressor or classifier
                parameters and values as strict values or list of strict values. Values
                of lists will be tuned.
            objective (str, optional): Metric to be used to optimize the model and train
                it. Available options:
                - Classification: multi:softmax, multi:softprob
                - Binary classification: binary:logistic, binary:logitraw, binary:hinge
                - Regression: reg:squarederror, reg:squaredlogerror, reg:logistic,
                    reg:pseudohubererror, reg:absoluteerror, reg:quantileerror
                Others could be found here: https://xgboost.readthedocs.io/en/stable/parameter.html#learning-task-parameters
                Defaults to "multi:softmax" for classification and "reg:squarederror"
                for regression.
            metric (str, optional): Metric to be used to evaluate the models trained
                with different values of the parameters provided. Available options:
                - Classification: f1, accuracy, weightedPrecision, weightedRecall,
                    weightedTruePositiveRate, weightedFalsePositiveRate,
                    weightedFMeasure, truePositiveRateByLabel,
                    falsePositiveRateByLabel, precisionByLabel, recallByLabel,
                    fMeasureByLabel, logLoss, hammingLoss.
                - Binary classification: areaUnderROC, areaUnderPR, and those of the
                    normal classification.
                - Regression: rmse, mse, r2, mae, var
                Defaults to "f1" for classification and "rmse" for regression.
            cv_folds (int, optional): Number of folds used in cross-validation.
                Defaults to 5.

        Returns:
            Params: parameters used to fit the best performing model considering the
                chosen metric.
            Transformer: fitted model with best performance considering the chosen
                metric.
        """
        if objective is None and self.type == "regression":
            objective = "reg:squarederror"
        if metric is None:
            metric = "f1" if self.type == "classification" else "rmse"

        df = self._preprocess_dataframe(df)

        if self._are_params_tunable(params):
            self._estimator = get_estimator(
                type=self.type,
                label_col_name=self.label_col_name,
                objective=objective,
                num_workers=self.num_workers,
            )
            best_params, self._transformer = tune_parameters_cv(
                df=df,
                params=params,
                estimator=self._estimator,
                evaluator=get_evaluator(
                    metric=metric, label_col_name=self.label_col_name
                ),
                num_folds=cv_folds,
            )
        else:
            raise ValueError(
                "params parameters should include at least one list of possible values."
            )

        return best_params, self._transformer

    def predict(self, df: DataFrame) -> DataFrame:
        """It predicts targets provided a DataFrame without a target with the model
        already fitted.

        You have to first call the fit method to use this one.

        Args:
            df (DataFrame): DataFrame with features to be used for inference.

        Returns:
            DataFrame: DataFrame with rawPrediction, prediction, probability new
                columns in case of classification and only prediction for regression.
        """
        self._check_model_already_fitted()

        df = self._preprocess_dataframe(df)

        return predict(df=df, transformer=self._transformer)  # type: ignore

    def score(self, df: DataFrame, metric=None) -> float:
        """It scores the predicted results (those obtained with the predict method).

        Args:
            df (DataFrame): DataFrame with rawPrediction, prediction, probability
                columns obtained from the predict method.
            metric (str, optional): Metric to be used to evaluate the model.
                Available options:
                - Classification: f1, accuracy, weightedPrecision, weightedRecall,
                    weightedTruePositiveRate, weightedFalsePositiveRate,
                    weightedFMeasure, truePositiveRateByLabel,
                    falsePositiveRateByLabel, precisionByLabel, recallByLabel,
                    fMeasureByLabel, logLoss, hammingLoss.
                - Binary classification: areaUnderROC, areaUnderPR, and those of the
                    normal classification.
                - Regression: rmse, mse, r2, mae, var
                Defaults to "f1" for classification and "rmse" for regression.

        Raises:
            Exception: if you have not used the predict method before the score one.

        Returns:
            float: value of the chosen metric.
        """
        if metric is None:
            metric = "f1" if self.type == "classification" else "rmse"

        self._check_model_already_fitted()

        if (
            self.type == "classification"
            and (
                "prediction" not in df.columns
                or "rawPrediction" not in df.columns
                or "probability" not in df.columns
            )
        ) or (self.type == "regression" and "prediction" not in df.columns):
            raise Exception("You have to run predict method before scoring.")

        return score(
            df=df,
            evaluator=get_evaluator(metric=metric, label_col_name=self.label_col_name),
        )

    def get_parameters(self, with_doc: bool = False, with_none=True) -> Union[
        dict[
            str, Union[Union[str, float, int, bool], list[Union[str, float, int, bool]]]
        ],
        dict[str, list[Any]],
    ]:
        """Extracts parameters from the estimator used the last time to fit or tune the
        model.

        Args:
            with_doc (bool, optional): if you want also the documentation of each
                parameter to be included. Defaults to False.
            with_none (bool, optional): if you want also to include parameters set to
                None. Defaults to True.

        Raises:
            Exception: if no model has been fit or tuned.

        Returns:
            dict[
                str,
                Union[Union[str, float, int, bool], list[Union[str, float, int, bool]]]
            ]: dictionary with parameter's names as keys and list of paramters value and
                documentation as values of the dictionary.
        """
        if self._estimator:
            params = {}
            for param, param_value in self._estimator.extractParamMap().items():
                if with_doc:
                    if with_none or (not with_none and param_value is not None):
                        params[param.name] = [param_value, param.doc]
                else:
                    if with_none or (not with_none and param_value is not None):
                        params[param.name] = param_value
            return params
        else:
            raise Exception("You have to fit or tune a model first.")

    def get_feature_importance(
        self, spark: SparkSession, importance_type: str = "weight"
    ) -> DataFrame:
        """Calculates the importance for the target of each feature used by the model.

        When a categorical variable has been used the output features are the as many
        values as the original categorical variable had. For example if feature column
        "alphabet" can be "a", "b", "c". The resulting features names in the DataFrame
        returned from this method is "alphabet_a", "alphabet_b", "alphabet_c".

        Args:
            spark (SparkSession): used to create the resulting DataFrame.
            importance_type (str): Type of calculation used for importance:
            - weight: the number of times a feature is used to split the data across
                all trees.
            - gain: the average gain across all splits the feature is used in.
            - cover: the average coverage across all splits the feature is used in.
            - total_gain: the total gain across all splits the feature is used in.
            - total_cover: the total coverage across all splits the feature is used
                in.
            Default to "weight".

        Returns:
            DataFrame: A spark DataFrame with two columns: feature and importance where
                each row is a feature with its importance.
        """
        self._check_model_already_fitted()

        features_names_and_importances = get_features_importance(
            transformer=self._transformer,  # type: ignore
            encoded_features_cols_name=self._encoded_features_cols_name,  # type: ignore
            importance_type=importance_type,
        )

        return spark.createDataFrame(
            list(features_names_and_importances.items()), ["feature", "importance"]
        ).orderBy(F.desc("importance"))

    def save(self, path: str) -> None:
        """Saves the trained model and the encoders (if present) in the provided folder.

        Args:
            path (str): Folter path where to save.
        """
        dbfs_path_path = Path(path)
        dbfs_path_parts = []
        for part in dbfs_path_path.parts:
            if part not in ["DBFS", "dbfs", "/", "\\"]:
                dbfs_path_parts.append(part)

        path_path = Path("/".join(dbfs_path_parts))

        if self._transformer is not None:
            self._transformer.save(str(path_path / "transformer"))  # type: ignore
        if self._estimator is not None:
            self._transformer.save(str(path_path / "estimator"))  # type: ignore
        if self._string_indexer_model is not None:
            self._string_indexer_model.save(str(path_path / "string_indexer_model"))
        if self._one_hot_encoder_model is not None:
            self._one_hot_encoder_model.save(str(path_path / "one_hot_encoder_model"))

        transformer_temp = self._transformer
        estimator_temp = self._estimator
        string_indexer_model_temp = self._string_indexer_model
        one_hot_encoder_model_temp = self._one_hot_encoder_model

        self._transformer = None
        self._estimator = None
        self._string_indexer_model = None
        self._one_hot_encoder_model = None

        with open(dbfs_path_path / "training_helper.pickle", "wb") as f:
            pickle.dump(self, f)

        self._transformer = transformer_temp
        self._estimator = estimator_temp
        self._string_indexer_model = string_indexer_model_temp
        self._one_hot_encoder_model = one_hot_encoder_model_temp

    @staticmethod
    def load(path: str) -> TrainingHelper:
        """Loads the trained model and the encoders (if present) from the provided
        folder.

        Args:
            path (str): Folder path where to retrieve the models.

        Raises:
            FileNotFoundError: If the training_helper.pickle is not found inside the
                provided folder.

        Returns:
            TrainingHelper: initialized TrainingHelper object.
        """
        path_path = Path(path)

        training_helper_path = path_path / "training_helper.pickle"
        if training_helper_path.exists():
            with open(training_helper_path, "rb") as f:
                training_helper = pickle.load(f)
        else:
            raise FileNotFoundError(f"{training_helper_path} not found")

        transformer_path = path_path / "transformer"
        if transformer_path.exists():
            if training_helper.type == "classification":
                transformer = SparkXGBClassifierModel()  # type: ignore
            else:
                transformer = SparkXGBRegressorModel()  # type: ignore
            training_helper._transformer = transformer.load(str(transformer_path))

        string_indexer_model_path = path_path / "string_indexer_model"
        if string_indexer_model_path.exists():
            string_indexer_model: StringIndexerModel = StringIndexerModel()
            training_helper._string_indexer_model = string_indexer_model.load(
                str(string_indexer_model_path)
            )

        one_hot_encoder_model_path = path_path / "one_hot_encoder_model"
        if one_hot_encoder_model_path.exists():
            one_hot_encoder_model: OneHotEncoderModel = OneHotEncoderModel()
            training_helper._one_hot_encoder_model = one_hot_encoder_model.load(
                str(one_hot_encoder_model_path)
            )

        return training_helper

    def _are_params_tunable(self, params) -> bool:
        """ "Checks if the params should be tuned or not.

        Returns:
            bool: True if a tuning process has to be done.
        """
        if params is None:
            return False
        for param_value in params.values():
            if isinstance(param_value, list):
                return True
        return False

    def _check_model_already_fitted(self) -> None:
        """Checks if the model has been fitted. Otherwise raises an exception.

        Raises:
            ValueError: if the model is not fitted.
        """
        if self._transformer is None:
            raise ValueError("You should fit the model first.")

    def _preprocess_dataframe(self, df: DataFrame) -> DataFrame:
        """Preprocess the DataFrame managing unordered and oredered categorical
        features variables and applying vector assembling to store every feature in
        one vector feature column.

        oredered categorical features are managed with label encoding while unordered
        categorical features are managed with one hot encoding.

        Args:
            df (DataFrame): DataFrame to be preprocessed.

        Returns:
            DataFrame: Preprocessed DataFrame with added "features" column.
        """
        label_cols_encoding: dict[str, list[Union[str, int, float]]] = {}
        if (
            self.ordered_categ_features_cols_name
            and len(self.ordered_categ_features_cols_name) > 0
        ):
            df, self._string_indexer_model, label_cols_encoding = label_encode(
                df=df,
                cols_name=self.ordered_categ_features_cols_name,
                string_indexer_model=self._string_indexer_model,
            )

        one_hot_cols_encoding: dict[str, list[Union[str, int, float]]] = {}
        if (
            self.unordered_categ_features_cols_name
            and len(self.unordered_categ_features_cols_name) > 0
        ):
            (
                df,
                self._one_hot_encoder_model,
                self._string_indexer_model,
                one_hot_cols_encoding,
            ) = one_hot_encode(
                df=df,
                cols_name=self.unordered_categ_features_cols_name,
                string_indexer_model=self._string_indexer_model,
                one_hot_encoder_model=self._one_hot_encoder_model,
            )

        label_cols_encoding.update(one_hot_cols_encoding)
        self._cols_encoding = label_cols_encoding

        df, self._encoded_features_cols_name = vector_assemble(
            df=df, features_cols_name=self.features_cols_name
        )

        return df
