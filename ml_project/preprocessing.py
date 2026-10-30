import json
import pickle

import numpy as np
import pandas as pd

DROP_COLUMNS = ["id", "attack_cat"]
TARGET_COLUMN = "label"
CATEGORICAL_COLUMNS = ["proto", "service", "state"]

TOP_PROTOCOLS = [
    "tcp",
    "udp",
    "unas",
    "arp",
    "ospf",
    "sctp",
    "any",
    "gre",
    "sun-nd",
    "swipe",
    "mobile",
    "pim",
    "ipv6",
    "rsvp",
    "sep",
]

LOG_TRANSFORM_COLUMNS = [
    "trans_depth",
    "response_body_len",
    "sbytes",
    "sloss",
    "dloss",
    "spkts",
    "dbytes",
    "dpkts",
    "dinpkt",
    "djit",
    "ct_flw_http_mthd",
    "sjit",
    "ct_ftp_cmd",
    "sload",
    "sinpkt",
    "dur",
    "synack",
    "ackdat",
    "dload",
    "tcprtt",
    "smean",
    "rate",
    "dmean",
    "ct_src_dport_ltm",
    "ct_dst_ltm",
    "ct_src_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_srv_dst",
    "ct_srv_src",
    "stcpb",
    "dtcpb",
]

OTHER_TOKEN = "__OTHER__"


class Preprocessor:
    def __init__(self):
        self.categorical_vocab = {}
        self.numeric_columns = []
        self.log_columns = list(LOG_TRANSFORM_COLUMNS)
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = []
        self.is_fitted = False

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()
        df = df.dropna(subset=[TARGET_COLUMN])
        for col in DROP_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        return df

    def _bucket_proto(self, series: pd.Series) -> pd.Series:
        return series.where(series.isin(TOP_PROTOCOLS), OTHER_TOKEN)

    def _one_hot(self, series: pd.Series, vocab: list, prefix: str) -> pd.DataFrame:
        safe = series.where(series.isin(vocab), OTHER_TOKEN)
        out = pd.DataFrame(
            0,
            index=series.index,
            columns=[f"{prefix}={v}" for v in vocab],
            dtype=np.float64,
        )
        for v in vocab:
            out[f"{prefix}={v}"] = (safe == v).astype(np.float64)
        return out

    def fit(self, train_df: pd.DataFrame):
        df = self._clean(train_df)
        proto_bucketed = self._bucket_proto(df["proto"])
        self.categorical_vocab["proto"] = TOP_PROTOCOLS + [OTHER_TOKEN]
        self.categorical_vocab["service"] = sorted(df["service"].unique().tolist()) + [
            OTHER_TOKEN
        ]
        self.categorical_vocab["state"] = sorted(df["state"].unique().tolist()) + [
            OTHER_TOKEN
        ]
        self.numeric_columns = [
            c for c in df.columns if c not in CATEGORICAL_COLUMNS and c != TARGET_COLUMN
        ]

        numeric = df[self.numeric_columns].astype(np.float64).copy()
        for col in self.log_columns:
            if col in numeric.columns:
                numeric[col] = np.log1p(numeric[col].clip(lower=0))

        self.mean_ = numeric.mean(axis=0).values.copy()
        self.std_ = numeric.std(axis=0).values.copy()
        self.std_[self.std_ == 0] = 1.0
        self.feature_names_ = []
        for col in CATEGORICAL_COLUMNS:
            for v in self.categorical_vocab[col]:
                self.feature_names_.append(f"{col}={v}")
        self.feature_names_.extend(self.numeric_columns)

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit() before transform().")
        df = self._clean(df)

        parts = [self._one_hot(df["proto"], self.categorical_vocab["proto"], "proto")]
        parts.append(
            self._one_hot(df["service"], self.categorical_vocab["service"], "service")
        )
        parts.append(
            self._one_hot(df["state"], self.categorical_vocab["state"], "state")
        )

        numeric = df[self.numeric_columns].astype(np.float64).copy()
        for col in self.log_columns:
            if col in numeric.columns:
                numeric[col] = np.log1p(numeric[col].clip(lower=0))
        numeric = (numeric.values - self.mean_) / self.std_
        numeric_df = pd.DataFrame(numeric, index=df.index, columns=self.numeric_columns)
        parts.append(numeric_df)

        X = pd.concat(parts, axis=1)[self.feature_names_].values.astype(np.float64)
        y = df[TARGET_COLUMN].values.astype(np.float64)
        return X, y

    def fit_transform(self, train_df: pd.DataFrame):
        self.fit(train_df)
        return self.transform(train_df)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, "rb") as f:
            obj.__dict__.update(pickle.load(f))
        return obj


def load_raw(train_path: str, test_path: str):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_raw(
        "data/UNSW_NB15_training-set.csv", "data/UNSW_NB15_testing-set.csv"
    )
    pre = Preprocessor()
    X_train, y_train = pre.fit_transform(train_df)
    X_test, y_test = pre.transform(test_df)

    print("X_train:", X_train.shape, "y_train:", y_train.shape)
    print("X_test :", X_test.shape, "y_test :", y_test.shape)
    print("Num features:", len(pre.feature_names_))
    print("Any NaN in X_train?", np.isnan(X_train).any())
    print("Any NaN in X_test?", np.isnan(X_test).any())
    print("Train label balance:", y_train.mean())
    print("Test label balance:", y_test.mean())
