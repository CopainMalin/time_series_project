import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

from sktime.forecasting.compose import make_reduction
from statsmodels.tsa.statespace.sarimax import SARIMAX
import lightgbm as lgb
from sklearn.linear_model import Ridge

from src.analysis_tools import plot_statistical_eval

########################################## NAN IMPUTING TOOLS ##########################################


def nan_imputer(
    data: pd.DataFrame,
    colonne: str,
    imputing_mode: str = "kalman",
    window: int = 24,
) -> pd.DataFrame:
    """Impute les valeurs manquantes d'une colonne pandas numérique.

    Args:
        data (pd.DataFrame): Le dataset à imputer.
        colonne (str): La colonne du dataset à imputer.
        imputing_mode (str, optional): Le mode d'imputation, "kalman" ou "rolling_mean". Defaults to "kalman".
            Si le choix est "kalman", impute les données manquantes par un filtre de kalman encapsulant un AR saisonnier d'ordre : (p = 4, i = 1, q = 0, P = 1, I = 1, Q = 0, période saisonnière = 24).
            Si le choix est "rolling_mean", impute les données manquantes par une moyenne mobile classique de fenêtre spécifiée par le paramètre "window".
        window (int, optional): La fenêtre à spécifier si le mode d'imputation est "rolling_mean". Defaults to 24.

    Raises:
        ValueError: Si la colonne spécifée n'est pas numérique.
        ValueError: Si le choix de la méthode d'imputation n'est ni "kalman" ni "rolling_mean"

    Returns:
        pd.DataFrame: Le dataframe imputé par la méthode choisie.
    """
    if not (pd.api.types.is_numeric_dtype(data[colonne])):
        raise ValueError("La colonne spécifiée doit être numérique")
    match imputing_mode:
        case "rolling_mean":
            imputed = (
                data[colonne].rolling(min_periods=1, center=True, window=window).mean()
            )
        case "kalman":
            model = SARIMAX(
                endog=data[
                    colonne
                ].values,  # Passage en numpy pour éviter les warnings sur le formattage des dates
                order=(4, 1, 0),
                seasonal_order=(1, 1, 0, 24),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted_model = model.fit()
            imputed = fitted_model.predict()
        case _:
            raise ValueError(
                "Le choix de la méthode d'imputing est inconnu, il doit être 'kalman' ou 'rolling_mean'."
            )

    return np.where(data[colonne].isna(), imputed, data[colonne])


def dataset_nan_imputer(data: pd.DataFrame) -> pd.DataFrame:
    """Impute les valeurs manquantes des colonnes numériques d'un dataset pandas.

    Args:
        data (pd.DataFrame): le dataset à imputer.

    Returns:
        pd.Dataframe: le dataset à imputé.
    """
    imputed_data = data.copy()
    for colonne in imputed_data.columns:
        imputed_data[colonne] = nan_imputer(data, colonne, imputing_mode="kalman")
    return imputed_data


########################################## CROSS VALIDATION ##########################################


def compute_forecasts(y_train: np.ndarray, horizon: int) -> dict:
    """Fit un SARIMAX, un modèle Naïf, un Light GBM (direct et recursif) et un Ridge (direct et recursif) et prédit les prochains steps.

    Args:
        y_train (np.ndarray): Le jeu de train.
        horizon (int): L'horizon de prévision.

    Returns:
        dict: La prévision associée a chaque modèle.
    """
    # Creating models
    ridge_forecaster = make_reduction(Ridge(), strategy="recursive")
    ridge_forecaster_direct = make_reduction(Ridge(), strategy="direct")

    forecaster = make_reduction(
        lgb.LGBMRegressor(n_estimators=1500, max_depth=3),
        strategy="recursive",
    )

    forecaster_direct = make_reduction(
        lgb.LGBMRegressor(n_estimators=1500, max_depth=3),
        strategy="direct",
    )

    # Fitting / forecasting
    forecasts = dict()
    forecasts["SARIMAX"] = (
        SARIMAX(
            endog=y_train,
            enforce_stationarity=False,
            enforce_invertibility=False,
            order=(3, 1, 0),
            seasonal_order=(1, 1, 0, 5),
        )
        .fit()
        .forecast(int(horizon))
        .values
    )
    # Parameters chosen so that the MLE get reached
    forecasts["Naive"] = [y_train.iloc[-1]] * horizon
    forecasts["LGBM"] = (
        forecaster.fit(y_train).predict(np.arange(1, horizon + 1)).values
    )
    forecasts["LGBM direct"] = (
        forecaster_direct.fit(y_train, fh=np.arange(1, horizon + 1)).predict().values
    )
    forecasts["Ridge"] = (
        ridge_forecaster.fit(y_train).predict(np.arange(1, horizon + 1)).values
    )
    forecasts["Ridge direct"] = (
        ridge_forecaster_direct.fit(y_train, fh=np.arange(1, horizon + 1))
        .predict()
        .values
    )
    return forecasts


def cross_validation(y: np.array, save: bool = True) -> tuple[dict, dict, dict]:
    """ "Cross valide" plusieurs modèles en faisant à la fois varier la taille de l'horizon et pour chaque horizon, l'emplacement de la fenêtre d'apprentissage et de prévision.
    Attention, la fonction est très couteuse, les résultats ont donc étés enregistrés en pickle.

    Args:
        y (np.array): Le vecteur sur lequel faire la cross validation (la série temporelle à modéliser)/

    Returns:
        tuple[dict, dict, dict]: 3 dictionnaires relatifs a la moyenne des MAE pour chaque horizon, ainsi qu'au max et au min de la MAE de chaque modèle pour chaque horizon
    """
    # Cross validation (test de plusieurs horizon et de plusieurs fit par horizon)
    MAX_TEST_SIZE = int(0.2 * len(y))
    MODELS = [
        "Naive",
        "Ridge",
        "Ridge direct",
        "SARIMAX",
        "LGBM",
        "LGBM direct",
    ]

    errors = {model: list() for model in MODELS}
    maxs = {model: list() for model in MODELS}
    mins = {model: list() for model in MODELS}
    confidence_interval_sizes = list()

    shift_error = {model: list() for model in MODELS}
    shift_maxs = {model: 10e-5 for model in MODELS}
    shift_mins = {model: 10e5 for model in MODELS}

    for horizon in tqdm(np.arange(1, MAX_TEST_SIZE, 5)):
        confidence_interval_sizes.append(MAX_TEST_SIZE - horizon - 1)
        print(
            f"Number of shift (horizon = {horizon}) : {confidence_interval_sizes[-1]}"
        )
        for shift in np.arange(1, MAX_TEST_SIZE - horizon):
            y_train, y_test = (
                y[shift : -MAX_TEST_SIZE + shift :],
                y[-MAX_TEST_SIZE + shift :],
            )
            forecasts = compute_forecasts(y_train, horizon)
            for model in forecasts.keys():
                temp_shift_err = np.mean(np.abs(forecasts[model], y_test[:horizon]))
                shift_error[model].append(temp_shift_err)
                if temp_shift_err > shift_maxs[model]:
                    shift_maxs[model] = temp_shift_err
                if temp_shift_err < shift_mins[model]:
                    shift_mins[model] = temp_shift_err
                del temp_shift_err

        for model in errors.keys():
            errors[model].append(np.mean(shift_error[model]))
            maxs[model].append(shift_maxs[model])
            mins[model].append(shift_mins[model])

    if save:
        with open("../pickle/errors.pkl", "wb") as fp:
            pickle.dump(errors, fp)

        with open("../pickle/maxs.pkl", "wb") as fp:
            pickle.dump(maxs, fp)

        with open("../pickle/mins.pkl", "wb") as fp:
            pickle.dump(mins, fp)

    return errors, maxs, mins


########################################## MODELISATION ET EVALUATION ##########################################


def get_prevision_gb(
    data: np.array, forecast_horizon: int
) -> tuple[np.array, np.array, np.array]:
    """Génère la prévision quantile du modèle final. On passe par les quantiles pour avoir les IC (bien que biaisés).

    Args:
        data (np.array): La série temporelle à modéliser, elle sera fit par le modèle.
        forecast_horizon (int): L'horizon de prévision.

    Returns:
        tuple(np.array, np.array, np.array) : La borne inférieure, la prévision et la borne supérieure de l'IC.
    """
    for alpha in [0.1, 0.5, 0.9]:
        regressor = lgb.LGBMRegressor(
            n_estimators=1500, max_depth=3, objective="quantile", alpha=alpha
        )

        forecaster = make_reduction(
            regressor, strategy="direct"  # hyper-paramter to set recursive strategy
        )

        match alpha:
            case 0.1:
                lower_bound_gb = forecaster.fit(
                    data, fh=np.arange(1, 1 + forecast_horizon)
                ).predict()
            case 0.5:
                forecast_gb = forecaster.fit(
                    data, fh=np.arange(1, 1 + forecast_horizon)
                ).predict()
            case 0.9:
                upper_bound_gb = forecaster.fit(
                    data, fh=np.arange(1, 1 + forecast_horizon)
                ).predict()

    return lower_bound_gb.ravel(), forecast_gb.ravel(), upper_bound_gb.ravel()


def statistical_evaluation(
    pred: np.ndarray, y_test: np.ndarray, n_iter: int = 10000
) -> float:
    """Evalue statistiquement la performance du modèle en répondant à la question : "Quelle est la probabilité que j'observe le même résultat dans un contexte de hasard ?"
    ATTENTION : Les données doivent être stationnaires.

    Args:
        pred (np.ndarray): La prévision du modèle du jeu de test.
        y_test (np.ndarray): Le jeu de test.
        n_iter (int, optional): Le nombre de permutation à réaliser. Defaults to 10000.

    Returns:
        float: La probabilité d'observer cette RMSE dans un contexte de hasard.
    """
    rmse_list = list()
    for i in range(10000):
        if i == 0:
            rmse_list.append(np.sqrt(np.mean(np.square(pred - y_test))))
        else:
            permuted_y_test = np.random.permutation(y_test)
            rmse_list.append(np.sqrt(np.mean(np.square(pred - permuted_y_test))))

    rmse_list = np.array(rmse_list)

    plot_statistical_eval(rmse_list)

    return np.sum(rmse_list <= rmse_list[0]) / rmse_list.shape[0]
