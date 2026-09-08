import importlib.util
import unittest
from pathlib import Path


UTILS = Path(__file__).parents[1] / "scripts" / "modulation_plot_utils.py"
SPEC = importlib.util.spec_from_file_location("modulation_plot_utils", UTILS)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ModulationSeriesTests(unittest.TestCase):
    def test_prepends_day_zero_for_initial_interval_value(self):
        days, values = MODULE.modulation_series(
            {"interval_times": [30, 60], "interval_values": [0.1, 0.2, 0.3]}
        )

        self.assertEqual(days, [0, 30, 60])
        self.assertEqual(values, [0.1, 0.2, 0.3])

    def test_preserves_one_timestamp_per_value(self):
        days, values = MODULE.modulation_series(
            {"interval_times": [30, 60], "interval_values": [0.2, 0.3]}
        )

        self.assertEqual(days, [30, 60])
        self.assertEqual(values, [0.2, 0.3])

    def test_rejects_ambiguous_lengths_before_plotting(self):
        with self.assertRaisesRegex(ValueError, "got 3 values and 1 times"):
            MODULE.modulation_series(
                {"interval_times": [30], "interval_values": [0.1, 0.2, 0.3]}
            )


if __name__ == "__main__":
    unittest.main()
