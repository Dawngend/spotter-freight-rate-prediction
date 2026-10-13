"""Check fitting-window isolation and the restricted December inputs."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import FreightFeatures, air_miles


def training_rows():
    frame = pd.DataFrame({
        "pickup": ["A", "A", "A", "A"], "delivery": ["B", "B", "B", "B"],
        "pickup_lat": [35.] * 4, "pickup_lon": [-85.] * 4,
        "delivery_lat": [40.] * 4, "delivery_lon": [-85.] * 4,
        "equipment": ["Dry Van", "Dry Van", "Reefer", "Reefer"],
        "weight": [10000., 20000., 30000., -1.],
        "date": ["2025-01-01"] * 4, "market_index": [1., 1.2, np.nan, 1.4]})
    frame["distance"] = air_miles(frame) * 1.2
    return frame


class FeatureTests(unittest.TestCase):
    def test_transform_uses_training_medians_and_keeps_the_supplied_distance(self):
        fit = training_rows()
        features = FreightFeatures(market=True).fit(fit)
        future = fit.iloc[[0, 2]].copy()
        future["weight"] = [-7., np.nan]
        future["market_index"] = np.nan
        future["distance"] = air_miles(future) * 78.5
        result = features.transform(future)
        np.testing.assert_allclose(result.weight, [15000., 30000.])
        np.testing.assert_allclose(result.market_index, 1.2)
        np.testing.assert_allclose(result.distance_used, future.distance)
        np.testing.assert_allclose(result.distance_ratio, 10.0)
        self.assertEqual(result.distance_flag.tolist(), [1, 1])
        self.assertEqual(result.weight_negative.tolist(), [1, 0])
        self.assertEqual(result.weight_missing.tolist(), [1, 1])
        future["weight"] = 1e9
        features.transform(future)
        self.assertEqual(features.weight_medians["Dry Van"], 15000.)

    def test_december_ignores_provided_coordinates_and_signals(self):
        fit = training_rows()
        features = FreightFeatures(december=True).fit(fit)
        future = fit.iloc[[0]].copy()
        expected = features.transform(future)
        future["pickup_lat"] = 0.
        future["market_index"] = 900.
        future["quote_signal"] = 800.
        pd.testing.assert_frame_equal(features.transform(future), expected)
        for column in ["month", "week_of_year", "market_index", "quote_signal", "load_id", "pickup", "delivery"]:
            self.assertNotIn(column, expected)
        future["pickup"] = "Unseen City"
        self.assertFalse(features.transform(future).isna().any().any())


if __name__ == "__main__":
    unittest.main()
