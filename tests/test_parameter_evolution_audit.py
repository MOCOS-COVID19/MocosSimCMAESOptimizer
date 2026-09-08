import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "drawing-utilities" / "build_parameter_evolution_audit.py"
SPEC = importlib.util.spec_from_file_location("parameter_evolution_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ParameterEvolutionAuditTests(unittest.TestCase):
    def test_joins_history_identity_to_config_and_filters_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "horizon_6m" / "iter_2" / "cand_03"
            config_dir.mkdir(parents=True)
            config = {
                "transmission_probabilities": {"school": .1, "class": .2, "age_coupling_param": .3},
                **{name: {"params": {"interval_values": [.4, .5]}} for name in MODULE.VECTORS},
            }
            (config_dir / "config.json").write_text(json.dumps(config))
            history = [
                {"fit_months": 6, "stage": "horizon_6m", "iteration": 2, "candidate": 3, "status": "completed", "score": 7},
                {"fit_months": 3, "stage": "horizon_3m", "iteration": 1, "candidate": 1, "status": "ok", "score": 1},
            ]
            records = MODULE.build_records(history, root)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["scalars"]["age_coupling_param"], .3)
            self.assertEqual(records[0]["vectors"]["infection_modulation"], [.4, .5])

    def test_render_contains_requested_controls(self):
        html = MODULE.render([])
        for value in ("6", "9", "12"):
            self.assertIn(f'data-stage="{value}"', html)
        for name in MODULE.SCALARS:
            self.assertIn(f'data-scalar="{name}"', html)
        for name in MODULE.VECTORS:
            self.assertIn(f'value="{name}"', html)

    def test_handoff_audit_flags_reverted_predecessor_buckets(self):
        def record(stage, score, values, candidate=1):
            return {"stage": stage, "iteration": 1, "candidate": candidate,
                    "score": score, "vectors": {name: values for name in MODULE.VECTORS}}

        records = [
            record(6, 2.0, [.1, .2, .3, .4, .5, .6], candidate=1),
            record(6, 1.0, [.1, .2, .3, .7, .8, .9], candidate=2),
            record(9, 3.0, [.1, .2, .3, .4, .5, .6, .7, .8, .9]),
        ]
        audit = MODULE.build_handoff_audit(records)
        self.assertEqual(audit[0]["source_winner"], [1, 2])
        self.assertFalse(audit[0]["vectors"]["infection_modulation"]["matches"])
        self.assertEqual(audit[0]["vectors"]["infection_modulation"]["different_buckets"],
                         [4, 5, 6])


if __name__ == "__main__":
    unittest.main()
