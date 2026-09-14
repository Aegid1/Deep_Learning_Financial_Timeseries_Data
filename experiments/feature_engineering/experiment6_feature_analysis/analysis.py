import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as ss
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests


#color can be red, blue, yellow
def plot_values_with_datetime(column_name:str, company_name:str, first_n_rows_included: int, color:str, path: str):
    df = pd.read_csv(os.path.join(path, f"{company_name}.csv"), parse_dates=['Datetime'])
    plt.figure(figsize=(12, 10))
    plt.plot(df["Datetime"][:first_n_rows_included], df[column_name][:first_n_rows_included], label=column_name, color=color)
    plt.xlabel("Date")
    plt.ylabel(column_name)
    plt.title(f"{column_name}-Datetime")
    save_path = os.path.join(company_name, f"{column_name}_{first_n_rows_included}_rows.png")
    plt.savefig(save_path, format='png', dpi=300)
    plt.show()


def determine_best_lag_for_ar_model(company_name:str):
    df = pd.read_csv(os.path.join("..", "..", "data", "stocks", f"{company_name}.csv"), parse_dates=['Datetime'])
    train = df["Open"].iloc[:1500].copy().reset_index(drop=True)
    test = df["Open"].iloc[1500:].copy().reset_index(drop=True)

    for i in range(2, 25, 1):
        model = AutoReg(train, lags=i).fit()
        predictions = model.forecast(steps=len(test))
        preds = [float(x) for x in predictions]
        mae = mean_absolute_error(test, preds)
        print(f"mean absolute error with lag: {i} = {mae}")


def determine_granger_causality(company_name:str, column_name:str):
    df = pd.read_csv(os.path.join("..", "..", "data", "stocks", f"{company_name}.csv"), parse_dates=['Datetime'])
    data = df[["Open", column_name]].dropna().reset_index(drop=True)
    result = grangercausalitytests(data[["Open", column_name]], maxlag=25, verbose=True)
    pvalues = {}
    for lag, tests in result.items():
        ftest = tests[0]['ssr_ftest']
        pvalues[lag] = ftest[1]

    with open(os.path.join(company_name, f"granger_causality_{column_name}"), "w", encoding="utf-8") as f:
        for lag, pval in pvalues.items():
            f.write(f"Lag {lag}: p-Wert = {pval:.6f}\n")


# Pearson-correlation (linear connection)
def create_pearson_heatmap(dataframe, company_name:str):
    pearson_corr = dataframe.corr(method='pearson')
    mask = np.triu(np.ones_like(pearson_corr, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Pearson-correlationmatrix")
    plt.savefig(os.path.join(company_name, f"{company_name}_pearson_heatmap.png"), format='png', dpi=300)
    plt.show()

# Spearman-correlation (monotonous connection, also non-linear)
def create_spearman_corr(dataframe, company_name:str):
    spearman_corr = dataframe.corr(method='spearman')
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Spearman-correlationmatrix")
    plt.savefig(os.path.join(company_name, f"{company_name}_spearman_heatmap.png"), format='png', dpi=300)
    plt.show()

'''
Mutual Information:
Measures how much information one feature provides about another without being limited to linear relationships.
This is particularly useful if there are complex, non-linear relationships between the features.
'''
def create_mutual_information(dataframe, company_name:str):
    target = dataframe["Target"]
    X = dataframe.drop("Target", axis=1)
    mi = mutual_info_regression(X, target)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    mi_series.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Mutual Information der Features mit dem Target")
    plt.ylabel("Mutual Information (in nats)")
    plt.xlabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(os.path.join(company_name, f"{company_name}_mutual_information.png"), format='png', dpi=300)
    plt.show()


def cramers_v(confusion_matrix):
    """Berechnet Cramérs V basierend auf einer Kontingenztabelle"""
    chi2 = ss.chi2_contingency(confusion_matrix)[0]  # Chi-Quadrat-Statistik
    n = confusion_matrix.sum().sum()  # Anzahl der Datenpunkte
    r, k = confusion_matrix.shape  # Zeilen & Spalten der Tabelle
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))  # Cramérs V Formel


def create_cramers_v_heatmap(df):

    cols = df.columns
    corr_matrix = pd.DataFrame(np.zeros((len(cols), len(cols))), index=cols, columns=cols)
    for col1 in cols:
        for col2 in cols:
            if col1 == col2:
                corr_matrix.loc[col1, col2] = 1.0
            else:
                confusion_matrix = pd.crosstab(df[col1], df[col2])
                corr_matrix.loc[col1, col2] = cramers_v(confusion_matrix)

    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Cramérs V Korrelation Heatmap")
    plt.show()


#create_cramers_v_heatmap(df)
#create_mutual_information(df)
#create_pearson_heatmap(df)
#create_spearman_corr(df)
#plot_values_with_datetime("sentiment", "Airbus", 100, "blue", os.path.join("..", "..", "data", "stocks", f"{company_name}.csv"))
#plot_values_with_datetime("log_return", "Airbus", -1, "blue", os.path.join("..", "..", "data", "stocks_stationary"))
#plot_values_with_datetime("economy_impact", "Airbus", 100, "red", os.path.join("..", "..", "data", "stocks", f"{company_name}.csv"))
#determine_best_lag_for_ar_model("BMW")
#determine_granger_causality("Siemens", "short_term_impact")
#determine_granger_causality("Siemens", "long_term_impact")