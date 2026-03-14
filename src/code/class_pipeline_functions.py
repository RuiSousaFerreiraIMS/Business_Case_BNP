import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    RobustScaler,
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.preprocessing import TargetEncoder



class ClientDataCleaner(BaseEstimator, TransformerMixin):
    """Handles initial data preparation: type conversion, column removal, and sanity checks.

    This transformer applies domain-driven rules defined at instantiation time.
    It does not learn anything from the data, so fit() is a no-op.

    Invalid numeric values are set to NaN (never dropped) so the imputer
    can handle them in the next pipeline step.

    Args:
        datetime_cols: Columns to convert to datetime.
        cols_to_remove: Columns to drop before any further processing.
        datetime_format: Optional strftime format string (e.g. '%Y-%m-%d').
            If None, pandas infers the format automatically.
        numeric_ranges: Dict mapping column name to (min, max) tuple.
            Values outside the range are set to NaN.
            Use None for no lower or upper bound e.g. (0, None).
        categorical_maps: Dict mapping column name to a normalisation dict
            e.g. {"gender": {"m": "male", "masculino": "male"}}.
            Unrecognised values are set to NaN.
        verbose: If True, prints a summary after each transform call.

    Attributes:
        report_: Dict with a summary of changes made during the last
            transform() call. Populated after transform() is called.
    """

    def __init__(
        self,
        datetime_cols: list[str] | None = None,
        cols_to_remove: list[str] | None = None,
        datetime_format: str | None = None,
        numeric_ranges: dict[str, tuple] | None = None,
        categorical_maps: dict[str, dict] | None = None,
        verbose: bool = False, ##Just for some logs
    ):
        self.datetime_cols    = datetime_cols
        self.cols_to_remove   = cols_to_remove
        self.datetime_format  = datetime_format
        self.numeric_ranges   = numeric_ranges
        self.categorical_maps = categorical_maps
        self.verbose          = verbose

    def fit(self, X, y=None):
        """No-op. This transformer uses only domain rules, not data statistics.

        Args:
            X: Input DataFrame or array. Ignored.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """Apply cleaning rules to X.

        Args:
            X: Input DataFrame or array with raw client data.

        Returns:
            Cleaned DataFrame with the same index as X.
        """
        df = pd.DataFrame(X).copy()
        self.report_ = {
            "removed_cols":      [],
            "converted_dates":   [],
            "numeric_new_nans":  {},
            "categorical_changes": {},
        }

        # --- Drop irrelevant columns ---
        if self.cols_to_remove:
            existing = [col for col in self.cols_to_remove if col in df.columns]
            df = df.drop(columns=existing)
            self.report_["removed_cols"] = existing

        # --- Convert to datetime ---
        if self.datetime_cols:
            for col in self.datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(
                        df[col],
                        format=self.datetime_format,
                        errors="coerce",  # invalid dates become NaT, not a crash
                    )
                    self.report_["converted_dates"].append(col)

        # --- Numeric sanity checks: out-of-range values -> NaN ---
        if self.numeric_ranges:
            for col, (low, high) in self.numeric_ranges.items():
                if col not in df.columns:
                    continue
                before_na = df[col].isna().sum()
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if low  is not None:
                    df.loc[df[col] < low,  col] = np.nan
                if high is not None:
                    df.loc[df[col] > high, col] = np.nan
                new_nans = int(df[col].isna().sum() - before_na)
                self.report_["numeric_new_nans"][col] = new_nans

        # --- Categorical normalisation: variants -> canonical label ---
        if self.categorical_maps:
            for col, mapping in self.categorical_maps.items():
                if col not in df.columns:
                    continue
                normalised = (
                    df[col].astype(str).str.strip().str.lower()
                )
                mapped = normalised.map(mapping)  # unrecognised values -> NaN
                n_changed = int((mapped != normalised).sum())
                df[col] = mapped
                self.report_["categorical_changes"][col] = n_changed

        if self.verbose:
            self._print_report(df)

        return df

    def _print_report(self, df):
        """Prints a human-readable summary of the last transform() call.

        Args:
            df: The transformed DataFrame, used to report final shape.
        """
        print("[ClientDataCleaner]")
        print(f"  Output shape     : {df.shape}")
        print(f"  Columns removed  : {self.report_['removed_cols'] or '—'}")
        print(f"  Dates converted  : {self.report_['converted_dates'] or '—'}")

        if self.report_["numeric_new_nans"]:
            print("  Numeric new NaNs :")
            for col, n in self.report_["numeric_new_nans"].items():
                print(f"    {col}: {n} new NaN(s)")

        if self.report_["categorical_changes"]:
            print("  Categorical fixes:")
            for col, n in self.report_["categorical_changes"].items():
                print(f"    {col}: {n} value(s) normalised")
        print()



class ClientOutlierHandler(BaseEstimator, TransformerMixin):
    """Detects and neutralises outliers using a multi-method voting approach.

    A value is only flagged as an outlier if at least `min_votes` detection
    methods agree. This conservative strategy avoids removing legitimate
    extreme clients (e.g. high earners, large loans) while still catching
    data entry errors.

    Outlier bounds are learned in fit() from the training fold only,
    preventing data leakage during cross-validation. Flagged values are
    set to NaN so the imputer in the next pipeline step handles them.
    Rows are never dropped.

    Supported detection methods:
        - 'iqr'   : Tukey fences — flags values below Q1 - k*IQR or
                    above Q3 + k*IQR.
        - 'mod_z' : Modified Z-score using median and MAD — more robust
                    than standard Z-score for skewed distributions like
                    income or loan amounts.

    Args:
        cols: Columns to check. If None, all numeric columns are used
            automatically, excluding id-like columns.
        methods: Detection methods to use. Any combination of 'iqr'
            and 'mod_z'.
        min_votes: Number of methods that must agree to flag a value.
            With 2 methods, setting min_votes=2 means both must agree —
            recommended to avoid removing legitimate extreme clients.
        iqr_k: Multiplier for the IQR fence. Standard is 1.5. Increase
            to 2.0-2.5 to be more conservative with tree-based models.
        z_thresh: Threshold for the modified Z-score. Standard is 3.5.
        skip_discrete: If True, skips columns with fewer unique values
            than discrete_unique_thresh. Avoids treating binary flags
            or ordinal codes as continuous distributions.
        discrete_unique_thresh: Max unique values for a column to be
            considered discrete and skipped.
        verbose: If True, prints a summary after each transform() call.

    Attributes:
        stats_: Dict mapping each column to its learned bounds per method.
            Populated after fit().
        cols_: List of columns that will be processed in transform().
            Populated after fit().
        report_: DataFrame with outlier counts per column from the last
            transform() call. Populated after transform().
        n_outliers_total_: Total number of cells flagged in the last
            transform() call. Populated after transform().

    """

    def __init__(
        self,
        cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        methods: tuple = ("iqr", "mod_z"),
        min_votes: int = 2,
        iqr_k: float = 1.5,
        z_thresh: float = 3.5,
        skip_discrete: bool = True,
        discrete_unique_thresh: int = 10,
        verbose: bool = False,
    ):
        self.cols                   = cols
        self.exclude_cols           = exclude_cols or []
        self.methods                = methods
        self.min_votes              = min_votes
        self.iqr_k                  = iqr_k
        self.z_thresh               = z_thresh
        self.skip_discrete          = skip_discrete
        self.discrete_unique_thresh = discrete_unique_thresh
        self.verbose                = verbose

    def fit(self, X, y=None):
        """Learn outlier bounds from training data.

        Args:
            X: Training DataFrame. Bounds are computed only on this data.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        df = pd.DataFrame(X)

        # --- Decide which columns to process ---
        if self.cols is None:
            candidates = df.select_dtypes(include="number").columns.tolist()
            # Skip id-like columns — outlier detection on IDs makes no sense
            self.cols_ = [
                c for c in candidates
                if not (str(c).lower().endswith("id") or str(c).lower() == "id")
            ]
        else:
            self.cols_ = [c for c in self.cols if c in df.columns]

        # Remove explicitly excluded columns (e.g. discrete codes like CSP)
        if self.exclude_cols:
            self.cols_ = [c for c in self.cols_ if c not in self.exclude_cols]

        self.stats_ = {}

        for col in self.cols_:
            s = pd.to_numeric(df[col], errors="coerce").dropna()

            if s.empty:
                self.stats_[col] = {}
                continue

            # Skip binary flags, ordinal codes, and other low-cardinality columns
            if self.skip_discrete and s.nunique() <= self.discrete_unique_thresh:
                self.stats_[col] = {}
                continue

            col_stats = {}

            # --- IQR fences (Tukey) ---
            if "iqr" in self.methods:
                q1  = float(s.quantile(0.25))
                q3  = float(s.quantile(0.75))
                iqr = q3 - q1
                col_stats["iqr"] = {
                    "lower": q1 - self.iqr_k * iqr,
                    "upper": q3 + self.iqr_k * iqr,
                }

            # --- Modified Z-score (median + MAD) ---
            if "mod_z" in self.methods:
                med = float(s.median())
                mad = float(np.median(np.abs(s - med)))

                if mad <= 0:
                    # MAD = 0 means no spread — skip this method for this column
                    col_stats["mod_z"] = {"lower": -np.inf, "upper": np.inf}
                else:
                    delta = (self.z_thresh * mad) / 0.6745
                    col_stats["mod_z"] = {
                        "lower": med - delta,
                        "upper": med + delta,
                    }

            self.stats_[col] = col_stats

        return self

    def transform(self, X):
        """Flag outliers as NaN using bounds learned in fit().

        Args:
            X: Input DataFrame. May be validation or test data.

        Returns:
            DataFrame with outliers replaced by NaN. Shape is unchanged.
        """
        check_is_fitted(self, "stats_")
        df = pd.DataFrame(X).copy()

        outliers_per_col = {}

        for col in self.cols_:
            if col not in df.columns:
                continue

            # Columns skipped during fit() have empty stats dicts
            if not self.stats_.get(col):
                outliers_per_col[col] = 0
                continue

            s     = pd.to_numeric(df[col], errors="coerce")
            votes = np.zeros(len(df), dtype=int)

            if "iqr" in self.stats_[col]:
                bounds  = self.stats_[col]["iqr"]
                votes  += ((s < bounds["lower"]) | (s > bounds["upper"])).astype(int)

            if "mod_z" in self.stats_[col]:
                bounds  = self.stats_[col]["mod_z"]
                votes  += ((s < bounds["lower"]) | (s > bounds["upper"])).astype(int)

            outlier_mask = votes >= self.min_votes
            n_flagged    = int(outlier_mask.sum())
            outliers_per_col[col] = n_flagged

            if n_flagged > 0:
                # Set to NaN — imputer handles these in the next step
                df.loc[outlier_mask, col] = np.nan

        # --- Build report ---
        self.report_ = (
            pd.DataFrame({
                "column":        list(outliers_per_col.keys()),
                "cells_flagged": list(outliers_per_col.values()),
            })
            .sort_values("cells_flagged", ascending=False)
            .reset_index(drop=True)
        )
        self.n_outliers_total_ = int(self.report_["cells_flagged"].sum())

        if self.verbose:
            self._print_report()

        return df

    def _print_report(self):
        """Prints a human-readable summary of the last transform() call."""
        print("[ClientOutlierHandler]")
        print(f"  Methods      : {self.methods}  (min_votes={self.min_votes})")
        print(f"  IQR k        : {self.iqr_k}  |  Z threshold : {self.z_thresh}")
        print(f"  Total flagged: {self.n_outliers_total_} cell(s) → set to NaN")
        print()
        active = self.report_[self.report_["cells_flagged"] > 0]
        if not active.empty:
            for _, row in active.iterrows():
                print(f"    {row['column']:<35} {row['cells_flagged']} outlier(s)")
        else:
            print("    No outliers detected.")
        print()

class ClientImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using statistics learned from training data only.

    Numeric columns are imputed using mean or median computed on the training
    fold. Datetime columns use forward or backward fill. Categorical columns
    are filled with a fixed placeholder string.

    All statistics are learned in fit() from the training fold only, which
    prevents data leakage during cross-validation.

    Args:
        numeric_strategy: Strategy for numeric columns. One of 'mean' or 'median'.
        datetime_strategy: Strategy for datetime columns. One of 'ffill' or 'bfill'.
        categorical_fill: Placeholder string for missing categorical values.
        verbose: If True, prints a summary after each transform() call.

    Attributes:
        numeric_fill_values_: Dict mapping numeric column name to the learned
            fill value (mean or median from training data).
        report_: Dict with a summary of imputation applied during the last
            transform() call. Populated after transform() is called.

    """

    def __init__(
        self,
        numeric_strategy: str = "median",   # "mean" | "median"
        datetime_strategy: str = "ffill",   # "ffill" | "bfill"
        categorical_fill: str = "Unknown",
        verbose: bool = False,
    ):
        # NOTE: ffill/bfill for datetime columns is row-order-dependent.
        # After a shuffled train/test split, ffill fills from a random
        # neighbor — not meaningful.  This is safe when no datetime cols
        # are present (as in the current ABT), but should be revisited
        # if raw date columns are ever passed through the pipeline.
        self.numeric_strategy  = numeric_strategy
        self.datetime_strategy = datetime_strategy
        self.categorical_fill  = categorical_fill
        self.verbose           = verbose

    def fit(self, X, y=None):
        """Learn fill values from training data.

        Args:
            X: Training DataFrame. Statistics are computed only on this data.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        df = pd.DataFrame(X)

        ## Learn numeric fill values from training fold only
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        self.numeric_fill_values_ = {}

        for col in numeric_cols:
            if self.numeric_strategy == "mean":
                self.numeric_fill_values_[col] = df[col].mean()
            elif self.numeric_strategy == "median":
                self.numeric_fill_values_[col] = df[col].median()
            else:
                raise ValueError(
                    f"Unknown numeric_strategy '{self.numeric_strategy}'. "
                    "Use 'mean' or 'median'."
                )

        ## Store column types seen during training
        ## Needed to apply the same logic consistently in transform()
        self.datetime_cols_    = df.select_dtypes(include="datetime").columns.tolist()
        self.categorical_cols_ = df.select_dtypes(include=["object", "category"]).columns.tolist()

        return self

    def transform(self, X):
        """Apply learned imputation to X.

        Args:
            X: Input DataFrame. May include validation or test data.

        Returns:
            DataFrame with no missing values.
        """
        df = pd.DataFrame(X).copy()
        self.report_ = {
            "numeric_imputed":     [],
            "datetime_imputed":    [],
            "categorical_imputed": [],
        }

        ## Numeric: use fill values learned in fit()
        for col, fill_value in self.numeric_fill_values_.items():
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(fill_value)
                self.report_["numeric_imputed"].append(col)

        ## Datetime: directional fill (no statistics needed)
        for col in self.datetime_cols_:
            if col in df.columns and df[col].isnull().any():
                if self.datetime_strategy == "ffill":
                    df[col] = df[col].ffill()
                elif self.datetime_strategy == "bfill":
                    df[col] = df[col].bfill()
                self.report_["datetime_imputed"].append(col)

        ## Categorical: fixed placeholder
        for col in self.categorical_cols_:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(self.categorical_fill)
                self.report_["categorical_imputed"].append(col)

        ## Safety check: no NaNs should remain
        remaining_nans = df.isnull().sum().sum()
        if remaining_nans > 0:
            cols_with_nans = df.columns[df.isnull().any()].tolist()
            raise ValueError(
                f"Imputation incomplete: {remaining_nans} NaN(s) remain "
                f"in columns: {cols_with_nans}. "
                "This may happen if a column type changed between fit() and transform()."
            )

        if self.verbose:
            self._print_report()

        return df

    def _print_report(self):
        """Prints a human-readable summary of the last transform() call."""
        print("[ClientImputer]")
        print(f"  Numeric  imputed [{self.numeric_strategy}]  : "
              f"{self.report_['numeric_imputed'] or '—'}")
        print(f"  Datetime imputed [{self.datetime_strategy}] : "
              f"{self.report_['datetime_imputed'] or '—'}")
        print(f"  Categ.   imputed ['{self.categorical_fill}']  : "
              f"{self.report_['categorical_imputed'] or '—'}")
        print()


class ClientOneHotEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes designated categorical columns robustly.
    
    Learns categories during fit() to ensure consistent dummy columns during
    transform(), even if some categories are missing in the test/inference data.
    
    Args:
        cols: List of column names to encode. If None, defaults to ['sdem_SITFAM', 'sdem_HABITAT'].
        drop_first: Whether to drop the first category encoded to avoid collinearity.
        verbose: If True, prints a summary after each transform() call.
    """
    def __init__(
        self,
        cols: list[str] | None = None,
        drop_first: bool = False,
        verbose: bool = False,
    ):
        if cols is None:
            cols = ["sdem_SITFAM", "sdem_HABITAT"]
        self.cols = cols
        self.drop_first = drop_first
        self.verbose = verbose
        self.dummy_columns_ = []
        
    def fit(self, X: pd.DataFrame, y=None):
        X_df = pd.DataFrame(X)
        cols_to_fit = [c for c in self.cols if c in X_df.columns]
        if cols_to_fit:
            # Generate dummies on training to learn exact column structure
            dummies = pd.get_dummies(
                X_df[cols_to_fit], 
                columns=cols_to_fit, 
                drop_first=self.drop_first,
                dtype=int
            )
            self.dummy_columns_ = dummies.columns.tolist()
        return self
        
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "dummy_columns_")
        df = pd.DataFrame(X).copy()
        cols_to_transform = [c for c in self.cols if c in df.columns]
        
        if cols_to_transform:
            dummies = pd.get_dummies(
                df[cols_to_transform], 
                columns=cols_to_transform, 
                drop_first=self.drop_first,
                dtype=int
            )
            # Align with columns learned during fit()
            # Missing columns are filled with 0, extra columns are dropped
            dummies = dummies.reindex(columns=self.dummy_columns_, fill_value=0)

            # Reset indices to prevent NaN rows from mismatched index alignment
            df = df.reset_index(drop=True)
            dummies = dummies.reset_index(drop=True)

            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=cols_to_transform, inplace=True)
            
        if self.verbose:
            print("[ClientOneHotEncoder]")
            print(f"  Encoded cols : {cols_to_transform}")
            print(f"  New features : {len(self.dummy_columns_)}")
            print()
            
        return df


class ClientFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates domain-informed features for early settlement and churn prediction.

    Features are derived from four source tables (BDOSSTOTAL, CRC, CredScore,
    FAMA_LIGHT_) and are grouped into six categories:
        1. Contract Maturity & Repayment
        2. Payment Behaviour
        3. Financial Capacity
        4. Portfolio & Relationship
        5. Scoring & Risk / Demographics
        6. Composite Scores

    All statistics used in relative features are learned in fit() from the
    training fold only, preventing data leakage during cross-validation.
    Division-by-zero is handled throughout by replacing 0 denominators with
    NaN before dividing — the imputer upstream ensures no NaNs reach this step,
    but the guards are kept as a safety net.

    Args:
        include_composite: If True, builds EARLY_SETTLE_PROPENSITY and
            CHURN_RISK_COMPOSITE weighted composite scores.
        drop_date_cols: If True, drops raw date columns after extracting
            CONTRACT_AGE_MONTHS. Downstream ColumnTransformer cannot handle
            datetime dtype.
        verbose: If True, prints a summary after each transform() call.

    Attributes:
        feature_names_created_: List of new column names added during the
            last transform() call. Populated after transform().
    """

    _SCORE_MAP    = {chr(ord("A") + i): i + 1 for i in range(24)}  # A=1 ... X=24
    _HOUSING_MAP  = {"P": 3, "A": 2, "F": 2, "L": 1, "O": 0, "X": 0}

    def __init__(
        self,
        include_composite: bool = True,
        drop_date_cols: bool = True,
        verbose: bool = False,
    ):
        self.include_composite = include_composite
        self.drop_date_cols    = drop_date_cols
        self.verbose           = verbose

    def fit(self, X, y=None):
        """No statistics to learn — all features use row-level calculations.

        fit() is kept as a no-op here because all engineered features are
        deterministic row-level transformations (ratios, differences, maps).
        No group statistics are needed, so there is nothing that could
        introduce leakage.

        Args:
            X: Training DataFrame. Not used.
            y: Ignored. Present for sklearn API compatibility.

        Returns:
            self
        """
        return self

    def transform(self, X):
        """Compute all engineered features and append them to the DataFrame.

        Args:
            X: Input DataFrame with merged client data from all source tables.

        Returns:
            DataFrame with original columns plus all new engineered features.
            Raw date columns are dropped if drop_date_cols=True.
        """
        df = pd.DataFrame(X).copy()
        cols_before = set(df.columns)

        # ------------------------------------------------------------------ #
        # 1. CONTRACT MATURITY & REPAYMENT                                     #
        # ------------------------------------------------------------------ #

        # Fraction of principal already repaid — clipped to [0, 1]  DATA LEAKAGE
        #df["REPAYMENT_RATIO"] = (
        #    (df["MTFINO"] - df["CRD"]) / df["MTFINO"].replace(0, np.nan)
        #).clip(0, 1)

        # How many months the contract has been active  DATA LEAKAGE
        #df["CONTRACT_AGE_MONTHS"] = (
            #(pd.to_datetime(df["DPOS"]) - pd.to_datetime(df["DCREAT"])).dt.days / 30.44
        #).clip(lower=0)

        # Estimated months remaining: current balance / monthly instalment THIS IS NOT THE CURRENT REMAINING MONTHS
        #df["REMAINING_TERM_MONTHS"] = (
            #f["MTFIN"] / df["MENSALIDADE"].replace(0, np.nan)
        #).clip(lower=0)

        # Fraction of contract lifecycle elapsed (0 = just started, 1 = done)  DATA LEAKAGE
        #total_term = df["CONTRACT_AGE_MONTHS"] + df["REMAINING_TERM_MONTHS"]
        #df["LIFECYCLE_RATIO"] = (
            #df["CONTRACT_AGE_MONTHS"] / total_term.replace(0, np.nan)
        #).clip(0, 1)

        # Whether current instalment exceeds original (>1 = voluntary overpayment)  MENSALIDADE_CORR IS = MENSALIDADE
        #df["OVERPAYMENT_RATIO"] = (
            #df["MENSALIDADE_CORR"] / df["MENSALIDADE"].replace(0, np.nan)
        #).clip(lower=0)

        # ------------------------------------------------------------------ #
        # 2. PAYMENT BEHAVIOUR                                                 #
        # ------------------------------------------------------------------ #

        # Fraction of agreed regularisations paid  ONLY PERTIENENT WHEN EVALUATING CONTRACT BY CONTRACT
        #df["REGULARIZATION_COMPLETION"] = (
            #df["RANGPRO"] / df["DURDEG"].replace(0, np.nan)
        #).clip(0, 1)

        # Payment delays normalised by contract age
        #df["DELAY_INTENSITY"] = (
            #df["RANGCLI"] / df["CONTRACT_AGE_MONTHS"].replace(0, np.nan)
        #).clip(lower=0)

        # Risk trend: positive = improving, negative = deteriorating
        #df["RISK_TREND_3M"] = df["RISK"].apply(self._compute_risk_trend)

        # Number of risk-level changes in last 12 months
        #df["RISK_VOLATILITY_12M"] = df["RISK"].apply(self._compute_risk_volatility)

        # Consecutive months at current risk level
        #df["MONTHS_AT_CURRENT_RISK"] = df["RISK"].apply(self._compute_months_at_current_risk)

        # ------------------------------------------------------------------ #
        # 3. FINANCIAL CAPACITY                                                #
        # ------------------------------------------------------------------ #

        # Debt-to-income using total external monthly obligations (CRC)
        df["DTI_RATIO"] = (
            df["MT_MENSAL"] / df["RESSO"].replace(0, np.nan)
        ).clip(lower=0)

        # Total outstanding debt across all external credit types
        debt_cols   = ["DIVIDAS_CL", "DIVIDAS_CP", "DIVIDAS_AUTO", "DIVIDAS_HT"]
        present_d   = [c for c in debt_cols if c in df.columns]
        df["TOTAL_DEBT_BURDEN"] = df[present_d].fillna(0).sum(axis=1)

        # Binary flag: any overdue amount at external lenders
        overdue_cols = ["MONTVENC_CL", "MONTVENC_CP", "MONTVENC_AUTO", "MONTVENC_HT"]
        present_ov   = [c for c in overdue_cols if c in df.columns]
        df["EXTERNAL_OVERDUE_FLAG"] = (
            df[present_ov].fillna(0).sum(axis=1) > 0
        ).astype(int)

        # Monthly instalment as fraction of remaining capital
        df["INSTALMENT_TO_CAPITAL"] = (
            df["MENSALIDADE"] / df["CRD"].replace(0, np.nan)
        ).clip(lower=0)

        # Share of income remaining after instalment
        df["INCOME_BUFFER"] = (
            (df["RESSO"] - df["MENSALIDADE"]) / df["RESSO"].replace(0, np.nan)
        ).clip(-1, 1)

        # ------------------------------------------------------------------ #
        # 4. PORTFOLIO & RELATIONSHIP                                          #
        # ------------------------------------------------------------------ #

        # Flag: client holds more than one product
        df["MULTI_PRODUCT_FLAG"] = (
            (df["ALLBD_N_CL__N"].fillna(0) + df["ALLBD_N_CP__N"].fillna(0)) > 1
        ).astype(int)

        # Total currently active contracts
        df["ACTIVE_CONTRACT_COUNT"] = (
            df["ALLBD_A_CL__N"].fillna(0) + df["ALLBD_A_CP__N"].fillna(0)
        )

        # Age of oldest contract (relationship depth)
        df["RELATIONSHIP_TENURE"] = df["ALLBD_IDADE_MSA__N"]

        # Age of most recent contract (recency of engagement)
        df["NEW_CONTRACT_RECENCY"] = df["ALLBD_IDADE_MIN__N"]

        # Contract modification frequency per month of avg contract age
        df["EVENT_FREQUENCY"] = (
            df["ALLBD_N_events__N"] / df["ALLBD_IDADE_MEAN__N"].replace(0, np.nan)
        ).clip(lower=0)

        # ------------------------------------------------------------------ #
        # 5. EXTERNAL CREDIT                                                   #
        # ------------------------------------------------------------------ #

        # Share of credit portfolio in consumer lending
        df["CREDIT_CONCENTRATION"] = (
            df["COUNT_CL"] / df["COUNT_TOTAL"].replace(0, np.nan)
        ).clip(0, 1)

        # Binary flag: any historical write-offs at any lender
        writeoff_cols = ["MONTABATV_CL", "MONTABATV_CP", "MONTABATV_AUTO", "MONTABATV_HT"]
        present_wo    = [c for c in writeoff_cols if c in df.columns]
        df["WRITEOFF_HISTORY_FLAG"] = (
            df[present_wo].fillna(0).sum(axis=1) > 0
        ).astype(int)

        # Fraction of total external debt that is overdue
        total_overdue = df[present_ov].fillna(0).sum(axis=1)
        df["OVERDUE_TO_DEBT_RATIO"] = (
            total_overdue / df["TOTAL_DEBT_BURDEN"].replace(0, np.nan)
        ).clip(0, 1)

        # ------------------------------------------------------------------ #
        # 6. SCORING & RISK                                                    #
        # ------------------------------------------------------------------ #

        # Numeric encoding of behavioral score: A=1 (best) ... X=24, null=25
        df["BEHAVIORAL_SCORE_NUM"] = (
            df["kp_sqe"].str.upper().map(self._SCORE_MAP).fillna(25).astype(int)
        )

        # Divergence: good behavioral score but high current risk position
        df["SCORE_RISK_MISMATCH"] = (
            df["kp_sqe"].str.upper().isin(set("ABCDE"))
            & df["RISKA"].isin([3, 4, 5])   # adjust to your actual risk scale
        ).astype(int)

        # ------------------------------------------------------------------ #
        # 7. DEMOGRAPHICS                                                      #
        # ------------------------------------------------------------------ #

        # Age brackets: distinct churn and settlement patterns by life stage
        df["AGE_BUCKET"] = pd.cut(
            df["sdem_age"],
            bins=[0, 25, 35, 45, 55, 65, 120],
            labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"],
            right=False,
        ).astype(str)

        # Housing stability numeric proxy
        df["HOUSING_STABILITY_SCORE"] = (
            df["sdem_HABITAT"].str.upper().map(self._HOUSING_MAP).fillna(0)
        )

        # Partnered / married flag (dual income proxy)
        df["IS_PARTNERED"] = (
            df["sdem_SITFAM"].str.upper().isin(["C", "U"])
        ).astype(int)

        # ------------------------------------------------------------------ #
        # 8. COMPOSITE SCORES                                                  #
        # ------------------------------------------------------------------ #

        if self.include_composite:
            bs_norm = 1 - (df["BEHAVIORAL_SCORE_NUM"] - 1) / 24

            df["EARLY_SETTLE_PROPENSITY"] = (
                0.30 * df["REPAYMENT_RATIO"]
              + 0.20 * (1 - df["DTI_RATIO"].clip(0, 1))
              + 0.20 * df["LIFECYCLE_RATIO"]
              + 0.15 * df["OVERPAYMENT_RATIO"].clip(upper=2) / 2
              + 0.15 * bs_norm
            )

            active_norm = 1 - df["ACTIVE_CONTRACT_COUNT"].clip(upper=5) / 5
            rt_norm     = (-df["RISK_TREND_3M"] + 1) / 2

            df["CHURN_RISK_COMPOSITE"] = (
                0.25 * df["DELAY_INTENSITY"].clip(upper=1)
              + 0.20 * df["EXTERNAL_OVERDUE_FLAG"]
              + 0.20 * active_norm
              + 0.20 * rt_norm
              + 0.15 * df["DTI_RATIO"].clip(upper=1)
            )

        # --- Drop raw date columns --- ColumnTransformer cannot handle datetime
        if self.drop_date_cols:
            date_cols = ["DPOS", "DCREAT", "DATFIN", "D1FIN", "DCSP"]
            df = df.drop(columns=[c for c in date_cols if c in df.columns])

        # --- Report ---
        self.feature_names_created_ = sorted(set(df.columns) - cols_before)

        if self.verbose:
            self._print_report(df)

        return df

    # ------------------------------------------------------------------ #
    # RISK STRING HELPERS                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_risk_trend(risk_str: str) -> float:
        """Compare average risk of last 3 months vs prior 3 months.

        Positive = improving (lower risk), negative = deteriorating.

        Args:
            risk_str: 24-character risk position string (or integer missing leading zeros).

        Returns:
            Float trend score. 0.0 if string is too short or invalid.
        """
        # Strip string, drop any decimal (if float), and zero pad to 24 characters
        cleaned = str(risk_str).strip().split('.')[0].zfill(24)
        if len(cleaned) < 6:
            return 0.0
        try:
            recent = [int(c) for c in cleaned[-3:]  if c.isdigit()]
            prior  = [int(c) for c in cleaned[-6:-3] if c.isdigit()]
            if not recent or not prior:
                return 0.0
            return float(np.mean(prior) - np.mean(recent))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_risk_volatility(risk_str: str) -> int:
        """Count risk-position changes in the last 12 months.

        Args:
            risk_str: 24-character risk position string (or integer missing leading zeros).

        Returns:
            Integer count of changes. 0 if invalid.
        """
        # Strip string, drop any decimal, zero pad to 24 characters
        cleaned = str(risk_str).strip().split('.')[0].zfill(24)
        last_12 = cleaned[-12:]
        digits  = [c for c in last_12 if c.isdigit()]
        if len(digits) < 2:
            return 0
        return sum(1 for i in range(1, len(digits)) if digits[i] != digits[i - 1])

    @staticmethod
    def _compute_months_at_current_risk(risk_str: str) -> int:
        """Count consecutive trailing months at the current risk level.

        Args:
            risk_str: 24-character risk position string (or integer missing leading zeros).

        Returns:
            Integer count. 0 if invalid.
        """
        # Strip string, drop any decimal, zero pad to 24 characters
        cleaned = str(risk_str).strip().split('.')[0].zfill(24)
        digits = [c for c in cleaned if c.isdigit()]
        if not digits:
            return 0
        current = digits[-1]
        count   = 0
        for c in reversed(digits):
            if c == current:
                count += 1
            else:
                break
        return count

    def _print_report(self, df: pd.DataFrame):
        """Prints a summary of features created during transform().

        Args:
            df: The transformed DataFrame.
        """
        print("[ClientFeatureEngineer]")
        print(f"  Features created : {len(self.feature_names_created_)}")
        print(f"  Output shape     : {df.shape}")
        print(f"  New columns      :")
        for name in self.feature_names_created_:
            print(f"    + {name}")
        print()
