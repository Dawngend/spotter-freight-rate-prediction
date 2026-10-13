"""Fit freight feature statistics using only the current training window."""

import numpy as np
import pandas as pd


COORDS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
CHART_INPUTS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]


def air_miles(frame):
    lat1, lon1, lat2, lon2 = np.radians(frame[COORDS].to_numpy()).T
    h = np.sin((lat2 - lat1) / 2) ** 2
    h += np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return pd.Series(3958.7613 * 2 * np.arcsin(np.sqrt(np.clip(h, 0, 1))), index=frame.index)


def distance_bucket(distance):
    return pd.cut(distance, [0, 250, 500, 1000, 1500, np.inf],
                  labels=["0-250", "250-500", "500-1000", "1000-1500", "1500+"])


class FreightFeatures:
    def __init__(self, market=False, december=False):
        self.market = market
        self.december = december

    def fit(self, frame):
        cities = []
        for side in ["pickup", "delivery"]:
            cities.append(frame[[side, f"{side}_lat", f"{side}_lon"]]
                          .set_axis(["city", "lat", "lon"], axis=1))
        cities = pd.concat(cities).drop_duplicates()
        if cities.city.duplicated().any():
            raise ValueError("Conflicting training coordinates for a city")
        self.cities = cities.set_index("city")
        weight = frame.weight.mask(frame.weight < 0)
        self.weight_medians = weight.groupby(frame.equipment).median()
        self.weight_median = weight.median()
        if self.market:
            self.market_median = frame.market_index.median()
        self.numeric_medians = self._build(frame).select_dtypes("number").median()
        return self

    def _build(self, frame):
        frame = frame.copy()
        if self.december:
            frame = frame[CHART_INPUTS].copy()
            for side in ["pickup", "delivery"]:
                for coord in ["lat", "lon"]:
                    frame[f"{side}_{coord}"] = frame[side].map(self.cities[coord])
        result = frame[COORDS + ["equipment"]].copy()
        result["air_miles"] = air_miles(frame)
        ratio = frame.distance / result.air_miles
        # The supplied distance is kept as is. The 12 train rows with a ratio above 3 are all 70.0 mile loads
        # (the smallest distance in the data) that sold for about 200 to 230, so the distance is right and the
        # city coordinates are the approximate part. The ratio stays a feature, capped to keep ridge stable.
        result["distance_flag"] = (ratio > 3).astype(int)
        result["distance_used"] = frame.distance
        result["log_distance"] = np.log(result.distance_used)
        result["distance_ratio"] = ratio.clip(upper=10)
        result["weight_negative"] = (frame.weight < 0).astype(int)
        weight = frame.weight.mask(frame.weight < 0)
        result["weight_missing"] = weight.isna().astype(int)
        result["weight"] = weight.fillna(frame.equipment.map(self.weight_medians)).fillna(self.weight_median)
        date = pd.to_datetime(frame.date)
        result["day_of_week"] = date.dt.dayofweek
        result["day_of_month"] = date.dt.day
        if not self.december:
            result["week_of_year"] = date.dt.isocalendar().week.astype(int)
            result["month"] = date.dt.month
        if self.market:
            result["market_missing"] = frame.market_index.isna().astype(int)
            result["market_index"] = frame.market_index.fillna(self.market_median)
        return result

    def transform(self, frame):
        return self._build(frame).fillna(self.numeric_medians)
