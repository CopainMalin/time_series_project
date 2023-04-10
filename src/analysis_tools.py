########################################## LIBRAIRIES ##########################################
# Arrays
import numpy as np
import pandas as pd

# Analyse
from nolds import lyap_r, dfa, hurst_rs, sampen
from statsmodels.tsa.stattools import adfuller

# Graphiques
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import aquarel

# Thème graphique
theme = aquarel.load_theme("arctic_dark")
theme.apply()


########################################## PLOTTING DESCRIPTIVE TOOLS ##########################################


def plot_boxplot(
    data: pd.DataFrame,
    figsize: tuple = (14, 6),
    title: str = "Boxplot des différentes colonnes",
    threshold: int = 0,
) -> None:
    """Plot un boxplot des colonnes numériques d'un dataframe.

    Args:
        data (pd.DataFrame): Le dataframe à étudier.
        figsize (tuple, optional): Les dimensions de la figure. Defaults to (14, 8).
        title (str, optional): Le titre de la figure. Defaults to "Boxplot des différentes colonnes".
        threshold (int, optional): Un seuil à faire apparaitre sur le plot. Defaults to 0.
    """
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=13, fontweight="bold")
    sns.boxenplot(data, orient="v")
    if threshold != None:
        plt.axhline(y=threshold, color="orangered", lw=2)
    plt.xticks(rotation=45)
    plt.show()


def plot_columns(
    data: pd.DataFrame,
    title: str = "Variation des différentes mesures",
    COLORS: dict() = None,
    figsize=(15, 5),
) -> None:
    """Plot les colonnes numériques d'un dataframe.

    Args:
        data (pd.DataFrame): Le dataframe à étudier.
        figsize (tuple, optional): Les dimensions de la figure. Defaults to (15, 5).
        COLORS (dict, optional): Le dictionnaire associant chaque variable à une couleur. Defaults to None.
    """
    if not (COLORS):
        COLORS = {k: f"C{v}" for (v, k) in enumerate(data.columns)}

    fig, ax = plt.subplots(
        nrows=3, ncols=2, figsize=(15, 10), sharex=False, sharey=False
    )
    axes = ax.ravel()
    axes[-1].remove()
    for i, (colname, coldata) in enumerate(
        data.select_dtypes(include=np.number).items()
    ):
        coldata.plot(ax=axes[i], legend=True, color=COLORS[colname])
        axes[i].axhline(0, color="white")
        axes[i].tick_params(axis="x", labelsize=10)
        axes[i].legend(loc="upper right", fontsize=10)
        axes[i].set_xlabel("")

    plt.suptitle(title, fontweight="bold", fontsize=13)

    plt.show()


def estimate_gaussian(
    data: pd.DataFrame,
    figsize: tuple = (15, 8),
    title: str = "Estimation des distributions des mesures",
    colors: dict() = None,
) -> None:
    """Trace la densitée estimée de chaque variable numérique d'un dataset ainsi qu'une estimation par la loi normale

    Args:
        data (pd.DataFrame): le dataset à étudier.
        figsize (tuple, optional): La taille de la figure. Defaults to (15, 8).
        title (str, optional): Le titre de la figure. Defaults to "Estimation des distributions des mesures".
        colors (dict, optional): Le dictionnaire de couleurs. Defaults to None.
    """
    plt.figure(figsize=(15, 8))
    plt.suptitle(title, fontsize=16, fontweight="bold")

    if colors == None:
        colors = {k: f"C{v}" for (v, k) in enumerate(data.columns)}

    for i, colonne in enumerate(data.select_dtypes(include=np.number).columns):
        estimated_law = np.random.normal(
            loc=np.mean(data[colonne]), scale=np.std(data[colonne]), size=10000
        )
        plt.subplot(2, 3, i + 1)
        sns.kdeplot(data[colonne], color=colors[colonne], fill=True)
        sns.kdeplot(
            estimated_law,
            color="white",
            linestyle="dashed",
            label=r"N($\mu =$"
            + f"{np.mean(data[colonne]):.0f}"
            + r",$\sigma =$"
            + f"{np.std(data[colonne]):.0f}"
            + r")",
        )
        plt.title(f"{colonne}")
        plt.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


########################################## (AUTO)CORRELATION PLOTTING TOOLS ##########################################


def plot_correlations(
    dataset: pd.DataFrame,
    method: str = "kendall",
    figsize: tuple = (14, 8),
    title: str = "Tau de Kendall entre les mesures",
    cmap: str = "coolwarm",
) -> None:
    """Affiche les corrélations entre les colonnes d'un dataframe.

    Args:
        dataset (pd.DataFrame): Le dataframe à étudier.
        method (str, optional): La méthode de calcul des corrélations. "pearson" non recommandé pour les séries chronologiques car peut introduire des corrélations fallacieuses. Defaults to "kendall".
        figsize (tuple, optional): La taille de la figure à afficher. Defaults to (14, 8).
        title (str, optional): Le titre de la figure. Defaults to "Tau de Kendall entre les mesures".
        cmap (str, optional): La colormap à choisir. Defaults to "coolwarm".
    """
    corr = dataset.corr(numeric_only=True, method=method)

    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=figsize)

    plt.title(title, fontsize=13, fontweight="bold")

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        vmax=0.3,
        center=0,
        annot=True,
        fmt=".1f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )

    plt.grid(visible=False)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


def acf_pacf_plot(
    vector: np.array,
    global_title: str = "",
    acf_title: str = "ACF",
    pacf_title: str = "PACF",
) -> None:
    """Plot les autocorrélations et autocorrélations partielles d'un vecteur donné.

    Args:
        vector (1d np.array): Le vecteur à analyser.
        global_title (str, optional): Le titre global du graphique. Defaults to ''.
        acf_title (str, optional): Le titre du graphique de l'ACF. Defaults to 'ACF'.
        pacf_title (str, optional): Le titre du graphique du PACF. Defaults to 'PACF'.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(vector, ax=ax1, title=acf_title)
    plot_pacf(vector, ax=ax2, title=pacf_title, method="ywm")

    fig.suptitle(global_title, fontsize=16, y=1.08)
    plt.tight_layout()
    plt.show()


def plot_all_acf_pacf(data: pd.DataFrame):
    """Plot les acf/pacf pour toutes les colonnes d'un dataframe

    Args:
        data (pd.DataFrame): Le dataframe à analyser.
    """
    for colonne in data.columns:
        acf_pacf_plot(
            vector=data[colonne], global_title=f"Analyse de la colonne {colonne}"
        )


########################################## INDICATORS COMPUTATIONS TOOLS ##########################################


def interpret_adf(p_value: float, seuil: float = 0.5) -> str:
    """Interprète le test de Dickey-Fuller augmenté.

    Args:
        p_value (float): La p-valeur du test.
        seuil (float, optional): Le seuil de significativité. Defaults to .5.

    Returns:
        str: L'interpretation vis-à-vis de la stationnarité.
    """
    if p_value < seuil:
        return "Stationnaire"
    else:
        return "Non stationnaire"


def compute_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calcul certains indicateurs à partir des séries temporelles d'un dataset :
            - L'exposant de Lyapunov renseignant sur la tendance chaotique de la série
            - L'exposant de Hurst renseignant sur tendance à long terme de la série
            - Le "detrended fluctual analysis" jouant le même rôle que l'exposant de Hurst, mais plus robuste notamment aux changements de saisonnalité
            - Le "sample entropy" permettant de déterminer, entre deux séries, laquelle est la plus facilement modélisable
    Args:
        data (pd.DataFrame): le dataframe à étudier.

    Returns:
        pd.DataFrame: La synthèse des résultats pour chaque colonne du dataframe.
    """
    lyapunov_list = list()
    hurst_list = list()
    dfa_list = list()
    sampen_list = list()
    adf_results = list()

    # Calcul des indicateurs
    for colonne in data.columns:
        if pd.api.types.is_numeric_dtype(data[colonne]):
            lyapunov_list.append(lyap_r(data[colonne], min_tsep=24, lag=1030))
            hurst_list.append(hurst_rs(data[colonne]))
            dfa_list.append(dfa(data[colonne]))
            sampen_list.append(sampen(data[colonne]))
            stat, p, _, _, _, _ = adfuller(data[colonne])
            adf_results.append(interpret_adf(p))

    # Stockage des indicateurs dans un dataframe
    results = pd.DataFrame(index=data.select_dtypes(include=np.number).columns)
    results["Exposant de Lyapunov"] = lyapunov_list
    results["Exposant de Hurst"] = hurst_list
    results["Detrended Fluctual Analysis"] = dfa_list
    results["Sample Entropy"] = sampen_list
    results["Dickey-Fuller Augmenté"] = adf_results
    return results.T


########################################## MODEL EVALUATION TOOLS ##########################################


def plot_results_cross_val(
    errors: dict,
    MAX_TEST_SIZE: int,
    maxs: dict = None,
    mins: dict = None,
    plot_maxs_mins: bool = False,
    figsize: tuple = (15, 5),
    title: str = "Variation des performances des modèles en fonction de l'horizon",
    MODELS: list() = [
        "Naive",
        "Ridge",
        "Ridge direct",
        "SARIMAX",
        "LGBM",
        "LGBM direct",
    ],
) -> None:
    """Plot les résultats de la cross validation.

    Args:
        errors (dict): Le dictionnaire des MAE moyennes par modèle issues de la cross validation.
        MAX_TEST_SIZE (int): La taille maximum du test size lors de la cross validation (typiquement égale à la taille du test set).
        maxs (dict): Le dictionnaire des MAE max par modèle issues de la cross validation. Requis si "plot_maxs_mins" est True. Defaults to None.
        mins (dict): Le dictionnaire des MAE max par modèle issues de la cross validation. Requis si "plot_maxs_mins" est True. Defaults to None.
        plot_maxs_mins (bool, optional): Conditionne le plot des bornes sup et inf. Defaults to False.
        figsize (tuple, optional): La taille de la figure. Defaults to (15, 5).
        title (str, optional): Le titre. Defaults to "Variation des performances des modèles en fonction de l'horizon".
        MODELS (list, optional): La liste des modèles évalués. Defaults to [ "Naive", "Ridge", "Ridge direct", "SARIMAX", "LGBM", "LGBM direct", ].
    """
    plt.figure(figsize=figsize)
    for model in MODELS:
        plt.plot(
            np.arange(1, MAX_TEST_SIZE, 5),
            errors[model],
            label=f"{model}",
            marker="o",
            markeredgewidth=1,
            markersize=10,
            markerfacecolor="None",
        )
        if plot_maxs_mins:
            plt.fill_between(
                x=np.arange(1, MAX_TEST_SIZE, 5),
                y1=maxs[model],
                y2=mins[model],
                alpha=0.6,
            )  # Make plot unreadable so we don't plot them
    plt.title(title, fontsize=13, fontweight="bold")
    plt.legend(loc="upper right")
    plt.xlabel("Taille de l'horizon")
    plt.ylabel("Erreur moyenne en valeur absolue")
    plt.show()


def plot_forecast(
    pred: np.array,
    lower: np.array,
    upper: np.array,
    y_train: np.array,
    y_test: np.array,
    title: str = "Estimation vs réalité du volume de trioxyde de tungstène",
    figsize: tuple = (15, 5),
    compute_RMSE: bool = True,
) -> None:
    """Plot le forecast du modèle.

    Args:
        pred (np.array): La prévision de la moyenne.
        lower (np.array): Le prévision de l'IC inf.
        upper (np.array): Le prévision de l'IC sup.
        y_train (np.array): Le train set apprit.
        y_test (np.array): Le test set prédit.
        title (str, optional): Le titre. Defaults to "Estimation vs réalité du volume de trioxyde de tungstène".
        figsize (tuple, optional): La taille de la figure. Defaults to (15, 5).
        compute_RMSE (bool, optional): Conditionne le calcule et l'affichage de la RMSE. Defaults to True.
    """
    plt.figure(figsize=(15, 5))
    plt.title(title, fontsize=13, fontweight="bold")
    plt.plot(np.arange(len(y_train))[-250:], y_train[-250:], color="white")
    plt.plot(
        np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
        y_test,
        color="white",
        label="Test set",
        linestyle="dashed",
    )
    plt.plot(
        np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
        pred,
        label="Forecast LGBM",
        color="dodgerblue",
    )
    plt.fill_between(
        x=np.arange(y_train.shape[0], y_train.shape[0] + y_test.shape[0]),
        y1=lower,
        y2=upper,
        alpha=0.2,
        color="dodgerblue",
    )
    if compute_RMSE:
        print(f"RMSE : {np.sqrt(np.mean(np.square(pred-y_test))):.0f}")
        print(
            f"RMSE normalisée : {np.sqrt(np.mean(np.square(pred-y_test))/np.mean(y_test)):.2f}%"
        )
    plt.legend()
    plt.show()


def plot_statistical_eval(rmse_list: np.array) -> None:
    """Plot l'évaluation statistique de la performance du modèle (caratèrisée par la RMSE).

    Args:
        rmse_list (np.array): La liste des RMSE observée (1ere sur le jeu non permuté et les autres sur le jeu permuté).
    """
    plt.figure(figsize=(15, 5))
    pval = np.sum(rmse_list <= rmse_list[0]) / rmse_list.shape[0]
    weights = np.ones(rmse_list.shape[0]) / rmse_list.shape[0]
    plt.hist(
        [rmse_list[rmse_list >= rmse_list[0]], rmse_list],
        histtype="stepfilled",
        alpha=0.8,
        color=["dodgerblue", "orangered"],
        bins=100,
        label="p-valeur<%.3f" % pval,
        weights=[weights[rmse_list >= rmse_list[0]], weights],
    )
    plt.axvline(
        x=rmse_list[0], color="white", linewidth=2, label="Statistique observée"
    )
    plt.ylabel("Densité des RMSE obtenues", fontsize=12, labelpad=10)
    plt.title(
        "Probabilité que la RMSE obtenue soit observée dans un contexte de hasard",
        fontsize=13,
        fontweight="bold",
    )
    plt.xlabel("Valeur de la RMSE obtenue")
    plt.legend()
    plt.show()
