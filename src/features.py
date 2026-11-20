"""Ingestion, target construction and leakage-free feature engineering."""
import numpy as np
import pandas as pd

from . import config as C


def load_raw(path=None) -> pd.DataFrame:
    df = pd.read_csv(path or C.DATA_PATH)
    for c in C.DATE_COLS:
        df[c] = pd.to_datetime(df[c], format="%Y-%m-%d", errors="coerce")
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Raw Days_Past_Due only records 90+ day write-offs, so arrears are recomputed from dates."""
    df = df.copy()
    df["arrears_days"] = (df["Account_Status_Date"] - df["Installment_Due_Date"]).dt.days
    df[C.TARGET] = (df["arrears_days"] >= 1).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Every history feature uses cumulative sums/shifts that EXCLUDE the current loan."""
    G = C.CLIENT
    df = df.sort_values([G, "Loan_Open_Date"], kind="mergesort").reset_index(drop=True)
    g = df.groupby(G, sort=False)
    df["_rr"] = df["LATEST_PYMT_AMT"] / df["Loan_Installment_Amount"].replace(0, np.nan)

    def prior_mean(col):
        v = df[col]
        cs = v.fillna(0).groupby(df[G]).cumsum() - v.fillna(0)
        cc = v.notna().astype(int).groupby(df[G]).cumsum() - v.notna().astype(int)
        return cs / cc.replace(0, np.nan)

    def prior_sum(col):
        v = df[col]
        return v.fillna(0).groupby(df[G]).cumsum() - v.fillna(0)

    def prior_max(col):
        return df.groupby(G)[col].shift(1).groupby(df[G]).cummax()

    def prior_roll_mean(col, w=3):
        s = df.groupby(G)[col].shift(1)
        return s.groupby(df[G]).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)

    def recency_weighted(col, weights=(0.5, 0.3, 0.2)):
        num = pd.Series(0.0, index=df.index)
        den = pd.Series(0.0, index=df.index)
        for k, w in enumerate(weights, start=1):
            s = df.groupby(G)[col].shift(k)
            num += w * s.fillna(0)
            den += w * s.notna().astype(float)
        return num / den.replace(0, np.nan)

    # counts, tenure, repayment quality
    df["borrowing_frequency"] = g.cumcount()
    df["is_first_loan"] = (df["borrowing_frequency"] == 0).astype(int)
    df["tenure_days"] = (df["Loan_Open_Date"] - g["Loan_Open_Date"].transform("min")).dt.days
    df["days_since_last_loan"] = (df["Loan_Open_Date"] - g["Loan_Open_Date"].shift(1)).dt.days
    df["repayment_ratio"] = prior_mean("_rr")
    df["_rr2"] = df["_rr"] ** 2
    var = (prior_mean("_rr2") - prior_mean("_rr") ** 2).clip(lower=0)
    df["payment_consistency"] = 1.0 / (1.0 + np.sqrt(var).fillna(0))
    df["avg_prior_loan_amount"] = prior_mean("Original_Loan_Amount")

    # prior arrears history
    df["prior_delinq_rate"] = prior_mean(C.TARGET)
    df["prior_delinq_cnt"] = prior_sum(C.TARGET)
    df["prior_mean_arrears"] = prior_mean("arrears_days")
    df["prior_max_arrears"] = prior_max("arrears_days")
    df["last_arrears"] = g["arrears_days"].shift(1)
    df["last_delinq"] = g[C.TARGET].shift(1)
    df["recent_delinq_rate"] = prior_roll_mean(C.TARGET, 3)
    df["recent_mean_arrears"] = prior_roll_mean("arrears_days", 3)
    df["arrears_trend"] = df["recent_mean_arrears"] - df["prior_mean_arrears"]

    # streaks, recency, early margin, volatility
    last_flag = g[C.TARGET].shift(1)
    chg = last_flag.ne(last_flag.groupby(df[G]).shift(1))
    blk = chg.groupby(df[G]).cumsum()
    run_pos = df.groupby([df[G], blk]).cumcount() + 1
    df["ontime_streak"] = np.where(last_flag.eq(0), run_pos, 0)
    df["late_streak"] = np.where(last_flag.eq(1), run_pos, 0)
    df["recency_arrears"] = recency_weighted("arrears_days")
    df["recency_delinq"] = recency_weighted(C.TARGET)
    df["_early_days"] = (-df["arrears_days"]).clip(lower=0)
    df["prior_early_margin"] = prior_mean("_early_days")
    df["_paid_early"] = (df["arrears_days"] < 0).astype(int)
    df["share_paid_early"] = prior_mean("_paid_early")
    df["_ad2"] = df["arrears_days"] ** 2
    df["arrears_volatility"] = np.sqrt(
        (prior_mean("_ad2") - prior_mean("arrears_days") ** 2).clip(lower=0))

    # capacity & momentum
    df["credits_to_installment"] = (df["Total_Mpesa_Credits_3Months"]
                                    / df["Loan_Installment_Amount"].replace(0, np.nan))
    df["loan_growth"] = df["Original_Loan_Amount"] / g["Original_Loan_Amount"].shift(1).replace(0, np.nan)
    gap_prior = prior_mean("days_since_last_loan").replace(0, np.nan)
    df["gap_trend"] = df["days_since_last_loan"] / gap_prior

    # delay class, demographics, bureau, ratios, calendar
    r = df["prior_delinq_rate"]
    df["delay_risk_class"] = np.where(df["borrowing_frequency"] == 0, "New",
                             np.where(r.isna() | (r == 0), "Never",
                             np.where(r < 0.25, "Occasional",
                             np.where(r <= 0.60, "Frequent", "Chronic"))))
    df["clean_history"] = ((df["prior_delinq_cnt"].fillna(0) == 0)
                           & (df["borrowing_frequency"] > 0)).astype(int)
    df["age"] = ((df["Loan_Open_Date"] - df["Date_Of_Birth"]).dt.days / 365.25).round(1)
    df["age_band"] = pd.cut(df["age"], bins=[0, 25, 35, 45, 55, 120],
                            labels=["<25", "25-35", "35-45", "45-55", "55+"]).astype(str)
    df["grade_ord"] = df["CRB_Grade"].map(C.GRADE_ORD).fillna(4).astype(int)
    df["open_month"] = df["Loan_Open_Date"].dt.month
    df["open_quarter"] = df["Loan_Open_Date"].dt.quarter
    inc = df["Estimated_Income"].replace(0, np.nan)
    df["Estimated_Income"] = inc
    df["utilization_ratio"] = df["Original_Loan_Amount"] / df["Credit_Limit"].replace(0, np.nan)
    df["mpesa_net_flow"] = df["Total_Mpesa_Credits_3Months"] - df["Total_Mpesa_Debits_3Months"]
    df["loan_to_income"] = df["Original_Loan_Amount"] / inc
    df["installment_burden"] = df["Loan_Installment_Amount"] / inc
    df["mpesa_debit_ratio"] = (df["Total_Mpesa_Debits_3Months"]
                               / df["Total_Mpesa_Credits_3Months"].replace(0, np.nan))
    df["loan_vs_avg_prior"] = df["Original_Loan_Amount"] / df["avg_prior_loan_amount"].replace(0, np.nan)
    df["util_x_grade"] = df["utilization_ratio"] * df["grade_ord"]

    assert not [f for f in C.FEATURES if f in C.LEAKY], "leakage: a post-origination field is a feature"
    df[C.NUMS] = df[C.NUMS].replace([np.inf, -np.inf], np.nan)
    return df
