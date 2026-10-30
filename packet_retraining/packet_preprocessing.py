import pickle

import numpy as np
import pandas as pd


PACKET_FIELDS = ["proto", "dur", "spkts", "dpkts", "sbytes", "dbytes", "sttl", "dttl"]
NUMERIC_FIELDS = PACKET_FIELDS[1:]
LOG_FIELDS = ["dur", "spkts", "dpkts", "sbytes", "dbytes"]
OTHER_PROTO = "__OTHER__"


class PacketPreprocessor:
    def __init__(self):
        self.protocols = []
        self.mean_ = None
        self.std_ = None
        self.feature_names_ = []
        self.is_fitted = False

    def _prepare(self, df):
        missing = [name for name in PACKET_FIELDS if name not in df.columns]
        if missing:
            raise ValueError(f"Missing packet fields: {', '.join(missing)}")

        protocol = df["proto"].astype("string").str.lower().str.strip()
        if protocol.isna().any():
            raise ValueError("Protocol must be present for every flow")

        numeric = df[NUMERIC_FIELDS].apply(pd.to_numeric, errors="raise").astype(np.float64)
        values = numeric.to_numpy()
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Packet fields must be finite and nonnegative")

        for name in LOG_FIELDS:
            numeric[name] = np.log1p(numeric[name])
        return protocol, numeric

    def fit(self, df):
        protocol, numeric = self._prepare(df)
        self.protocols = protocol.value_counts().head(15).index.tolist()
        self.mean_ = numeric.mean(axis=0).to_numpy()
        self.std_ = numeric.std(axis=0).to_numpy()
        self.std_[self.std_ == 0] = 1.0
        self.feature_names_ = [f"proto={name}" for name in self.protocols + [OTHER_PROTO]]
        self.feature_names_.extend(NUMERIC_FIELDS)
        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("Fit the packet preprocessor before transforming data")

        protocol, numeric = self._prepare(df)
        protocol = protocol.where(protocol.isin(self.protocols), OTHER_PROTO)
        categories = self.protocols + [OTHER_PROTO]
        encoded = np.column_stack([(protocol == name).to_numpy(dtype=np.float64) for name in categories])
        scaled = (numeric.to_numpy() - self.mean_) / self.std_
        features = np.column_stack([encoded, scaled])

        labels = None
        if "label" in df.columns:
            labels = pd.to_numeric(df["label"], errors="raise").to_numpy(dtype=np.float64)
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("Labels must be 0 or 1")
        return features, labels

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def save(self, path):
        with open(path, "wb") as file:
            pickle.dump(self.__dict__, file)

    @classmethod
    def load(cls, path):
        item = cls()
        with open(path, "rb") as file:
            item.__dict__.update(pickle.load(file))
        return item
