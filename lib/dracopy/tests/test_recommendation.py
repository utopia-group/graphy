import unittest

from lib.dracopy.draco.helper import data_to_asp
from lib.dracopy.draco.utils import cql_to_asp
from lib.dracopy.draco.run import run


def get_rec(data_schema, spec, relax_hard=False):
    query = cql_to_asp(spec)
    return run(data_schema + query, relax_hard=relax_hard)


spec_schema = [
    'data("data.csv").',
    "num_rows(100).",
    'fieldtype("q1",number).',
    'cardinality("q1",100).',
    'entropy("q1",1).',
    'fieldtype("q2",number).',
    'cardinality("q2",100).',
    'entropy("q2",1).',
    'fieldtype("o1",number).',
    'cardinality("o1",6).',
    'entropy("o1",1).',
    'fieldtype("n1",string).',
    'cardinality("n1",5).',
    'entropy("n1",1).',
]


class TestSpecs:
    def test_scatter(self):
        recommendation = get_rec(
            spec_schema,
            {"encodings": [{"channel": "x", "field": "q1"}, {"field": "q2"}]},
        ).as_vl()

        assert recommendation == {
            "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
            "data": {"url": "data.csv"},
            "mark": "point",
            "encoding": {
                "x": {"field": "q1", "type": "quantitative", "scale": {"zero": True}},
                "y": {"field": "q2", "type": "quantitative", "scale": {"zero": True}},
            },
        }

    def test_histogram(self):
        recommendation = get_rec(
            spec_schema, {"encodings": [{"field": "q1", "bin": True, "channel": "x"}]}
        ).as_vl()

        print(recommendation)
        assert recommendation == {
            "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
            "data": {"url": "data.csv"},
            "mark": "bar",
            "encoding": {
                "x": {"field": "q1", "type": "quantitative", "bin": True},
                "y": {
                    "aggregate": "count",
                    "type": "quantitative",
                    "scale": {"zero": True},
                },
            },
        }

    def test_strip(self):
        recommendation = get_rec(spec_schema, {"encodings": [{"field": "q1"}]}).as_vl()

        assert recommendation == {
            "$schema": "https://vega.github.io/schema/vega-lite/v3.json",
            "data": {"url": "data.csv"},
            "mark": "tick",
            "encoding": {
                "x": {"field": "q1", "type": "quantitative", "scale": {"zero": True}}
            },
        }

    def test_disable_hard_integrity(self):
        recommendation = get_rec(
            spec_schema,
            {"encodings": [{"field": "n1", "scale": {"log": True}}]},
            relax_hard=True,
        )
        assert recommendation is not None


class TestTypeChannel:
    def get_spec(self, t, channel):
        return {
            "mark": "point",
            "encoding": {
                "y": {"field": "q1", "type": "quantitative"},
                channel: {"field": "q2" if t == "quantitative" else "o1", "type": t},
            },
        }