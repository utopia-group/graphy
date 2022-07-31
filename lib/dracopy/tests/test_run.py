import json
import os

from jsonschema import validate

from lib.dracopy.draco.run import run
from lib.dracopy.draco.helper import read_data_to_asp
from lib.dracopy.draco.utils import cql_to_asp

EXAMPLES_DIR = os.path.join("examples")

class TestFull:
    def test_output_schema(self):
        json_files = [
            os.path.join(EXAMPLES_DIR, fname)
            for fname in os.listdir(EXAMPLES_DIR)
            if fname.endswith(".json") and not fname.endswith(".vl.json")
        ]

        with open(os.path.join("vega-lite-schema.json")) as sf:
            schema = json.load(sf)

            print(json_files)

            for fname in json_files:
                with open(fname, "r") as f:
                    query_spec = json.load(f)

                    data = None

                    if "url" in query_spec["data"]:
                        data = read_data_to_asp(
                            os.path.join(
                                os.path.dirname(f.name), query_spec["data"]["url"]
                            )
                        )
                    elif "values" in query_spec["data"]:
                        data = read_data_to_asp(query_spec["data"]["values"])
                    else:
                        raise Exception("no data found in spec")

                    query = cql_to_asp(query_spec)

                    program = query + data

                    print("-=====>")
                    results = run(program, multiple_solution=True)
                    result = results[-1]
                    print(program)
                    print(fname)
                    print(json.dumps(result.as_vl()))
                    print("<<<------")
                    validate(result.as_vl(), schema)
                    

# result = run(['data("data/cars.csv").', 'encoding(e0).', 'channel(e0,x).', 'field(e0,"horsepower").', 'num_rows(406).', 
#     'fieldtype("name",string).', 'cardinality("name",311).', 'fieldtype("miles_per_gallon",number).', 
#     'cardinality("miles_per_gallon",129).', 'fieldtype("cylinders",number).', 'cardinality("cylinders",5).', 
#     'fieldtype("displacement",number).', 'cardinality("displacement",83).', 'fieldtype("horsepower",number).', 
#     'cardinality("horsepower",93).', 'fieldtype("weight_in_lbs",number).', 'cardinality("weight_in_lbs",356).', 
#     'fieldtype("acceleration",number).', 'cardinality("acceleration",96).', 'fieldtype("year",string).', 
#     'cardinality("year",12).', 'fieldtype("origin",string).', 'cardinality("origin",3).'], multiple_solution=True)
# for l in result:
#     print(json.dumps(l.as_vl()))

# TestFull().test_output_schema()
