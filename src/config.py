"""Central configuration for the early-delinquency pipeline."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data" / "Digital_Banking_MSC_2024_2025_v3.csv"))
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", ROOT / "artifacts"))
BUNDLE_PATH = ARTIFACT_DIR / "early_delinquency_bundle.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

RANDOM = 42
TARGET = "early_delinquency"
CLIENT = "Unique_Client_Number"

# Business settings (substitute the institution's own economics)
PREC_FLOOR = 0.60
VALUE_TP = 4.0
COST_FP = 1.0
N_TRIALS = int(os.getenv("N_TRIALS", 40))

# Selected Optuna configuration (Table 7 of the report); used with --no-tune
BEST_PARAMS = dict(
    learning_rate=0.0172, num_leaves=146, min_child_samples=141, subsample=0.794,
    colsample_bytree=0.686, reg_alpha=1.175, reg_lambda=0.008, scale_pos_weight=1.644,
)

GRADE_ORD = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "X": 6, "R": 7}
DATE_COLS = ["Loan_Open_Date", "Installment_Due_Date", "Date_Of_Birth",
             "Latest_Payment_Date", "Loan_Disbursement_Date", "Account_Status_Date"]

FEATURES = [
    "Gender", "age", "age_band",
    "Original_Loan_Amount", "Credit_Limit", "utilization_ratio", "loan_to_income",
    "installment_burden", "loan_vs_avg_prior", "avg_prior_loan_amount", "util_x_grade",
    "credits_to_installment", "loan_growth",
    "tenure_days", "borrowing_frequency", "is_first_loan", "days_since_last_loan", "gap_trend",
    "repayment_ratio", "payment_consistency",
    "prior_delinq_rate", "prior_delinq_cnt", "prior_mean_arrears", "prior_max_arrears",
    "last_arrears", "last_delinq", "recent_delinq_rate", "recent_mean_arrears", "arrears_trend",
    "ontime_streak", "late_streak", "recency_arrears", "recency_delinq",
    "prior_early_margin", "share_paid_early", "arrears_volatility",
    "delay_risk_class", "clean_history",
    "Total_Mpesa_Credits_3Months", "mpesa_net_flow", "mpesa_debit_ratio", "Estimated_Income",
    "CRB_Grade", "grade_ord", "CRB_No_Of_Loans_Last_12Months", "CRB_Max_DPD_Last_1_Months",
    "CRB_Max_DPD_Last_2_Months", "CRB_MAX_Loans_Ever", "CRB_Days_Since_Last_Loan",
    "CRB_Max_Limit_Last_12_Months", "open_month", "open_quarter",
]
CATS = ["Gender", "age_band", "delay_risk_class", "CRB_Grade"]
NUMS = [c for c in FEATURES if c not in CATS]

# Fields resolved after origination: must never be model inputs
LEAKY = ["Days_Past_Due", "Overdue_Balance", "Installment_Arrears", "Current_Outstanding_Balance",
         "Account_Status", "Account_Status_Date", "Account_Status_Description", "Latest_Payment_Date",
         "LATEST_PYMT_AMT", "Installment_Due_Date", "arrears_days"]

# Client-level snapshot attributes: constant within every client in this extract (leakage audit)
SNAPSHOT_FEATURES = [
    "CRB_Grade", "grade_ord", "CRB_No_Of_Loans_Last_12Months", "CRB_Max_DPD_Last_1_Months",
    "CRB_Max_DPD_Last_2_Months", "CRB_MAX_Loans_Ever", "CRB_Days_Since_Last_Loan",
    "CRB_Max_Limit_Last_12_Months", "Total_Mpesa_Credits_3Months", "mpesa_net_flow",
    "mpesa_debit_ratio", "Estimated_Income", "util_x_grade", "credits_to_installment",
    "loan_to_income", "installment_burden",
]
