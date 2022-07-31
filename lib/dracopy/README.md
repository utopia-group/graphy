# Formalizing Visualization Design Knowledge as Constraints

Draco is a formal framework for representing design knowledge about effective visualization design as a collection of constraints. You can use Draco to find effective visualization visual designs in Vega-Lite. Draco's constraints are implemented in based on Answer Set Programming (ASP) and solved with the Clingo constraint solver. We also implemented a way to learn weights for the recommendation system directly from the results of graphical perception experiment.

Read our introductory [blog post about Draco](https://medium.com/@uwdata/draco-representing-applying-learning-visualization-design-guidelines-64ce20287e9d) and our [research paper](https://idl.cs.washington.edu/papers/draco/) for more details. Try Draco in the browser at https://uwdata.github.io/draco-editor.

## Status

**There Be Dragons!** This project is in active development and we are working hard on cleaning up the repository and making it easier to use the recommendation model in Draco. If you want to use this right now, please talk to us. More documentation is forthcoming.

## Overview

This repository currently contains:

* [**draco**](https://pypi.org/project/draco/) (pypi) The ASP programs with soft and hard constraints, a python API for [running Draco](https://github.com/uwdata/draco/blob/master/draco/run.py), the [CLI](https://github.com/uwdata/draco/blob/master/draco/cli.py), and the [python implementation](https://github.com/uwdata/draco/blob/master/draco/utils.py) for the **draco-core** API. Additionally includes some [helper functions](https://github.com/uwdata/draco/blob/master/draco/helper.py) that may prove useful.

### Sibling Repositories

Various functionality and extensions are in the following repositories

* [draco-vis](https://github.com/uwdata/draco-vis)
   * A web-friendly Draco! Including a bundled Webassembly module of Draco's solver, Clingo.

* [draco-learn](https://github.com/uwdata/draco-learn)
   * Runs a learning-to-rank method on results of perception experiments.
   
* [draco-tools](https://github.com/uwdata/draco-tools)
   * UI tools to create annotated datasets of pairs of visualizations, look at the recommendations, and to explore large datasets of example visualizations.
   
* [draco-analysis](https://github.com/uwdata/draco-analysis)
   * Notebooks to analyze the results.
   
## Draco API (Python)

In addition to a wrapper of the Draco-Core API describe below, the python API contains the following functions.

*object* **Result** [<>](https://github.com/uwdata/draco/blob/2de31e3eeb6eab29577b1b09a92ab3c0fd7bd2e0/draco/run.py#L36)

>The result of a Draco run, a solution to a draco_query. User `result.as_vl()` to convert this solution into a Vega-Lite specification.

**run** *(draco_query: List[str] [,constants, files, relax_hard, silence_warnings, debug, clear_cache]) -> Result:* [<>](https://github.com/uwdata/draco/blob/2de31e3eeb6eab29577b1b09a92ab3c0fd7bd2e0/draco/run.py#L115)

>Runs a `draco_query`, defined as a list of Draco ASP facts (strings), against given `file` asp programs (defaults to base Draco set). Returns a `Result` if the query is satisfiable. If `relax_hard` is set to `True`, hard constraints (`hard.lp`) will not be strictly enforced, and instead will incur an infinite cost when violated.

**is_valid** *(draco_query: List[str] [,debug]) -> bool:* [<>](https://github.com/uwdata/draco/blob/2de31e3eeb6eab29577b1b09a92ab3c0fd7bd2e0/draco/helper.py#L10)

>Runs a `draco_query`, defined as a list of Draco ASP facts (strings), against Draco's hard constraints. Returns true if the visualization defined by the query is a valid one (does not violate hard constraints), and false otherwise. Hard constraints can be found in [`hard.lp`](https://github.com/uwdata/draco/blob/master/asp/hard.lp).

**data_to_asp** *(data: List) -> List[str]:* [<>](https://github.com/uwdata/draco/blob/2de31e3eeb6eab29577b1b09a92ab3c0fd7bd2e0/draco/helper.py#L24)

>Reads an array of `data` and returns the ASP declaration of it (a list of facts).

**read_data_to_asp** *(file: str) -> List[str]:* [<>](https://github.com/uwdata/draco/blob/2de31e3eeb6eab29577b1b09a92ab3c0fd7bd2e0/draco/helper.py#L24)

>Reads a `file` of data (either `.json` or `.csv`) and returns the ASP declaration of it (a list of facts).

## User Info

### Installation

#### Python (Draco API)

##### Install Clingo

You can install Clingo with conda: `conda install -c potassco clingo`. On MacOS, you can alternatively run `brew install clingo`.

##### Install Draco (Python)

`pip install draco`

## Developer Info

### Installation

#### Install Clingo.

You can install Clingo with conda: `conda install -c potassco clingo`. On MacOS, you can alternatively run `brew install clingo`.

#### Python setup

`pip install -r requirements.txt` or `conda install --file requirements.txt`

Install Draco in editable mode. We expect Python 3.

`pip install -e .`

Now you can call the command line tool `draco`. For example `draco --version` or `draco --help`.

#### Tests

You should also be able to run the tests (and coverage report)

`python setup.py test`

##### Run only ansunit tests

`ansunit asp/tests.yaml`

##### Run only python tests

`pytest -v`

##### Test types

`mypy draco tests --ignore-missing-imports`

### Running Draco

#### Run Draco directly on a set of ASP constraints

You can use the helper file `asp/_all.lp`.

`clingo asp/_all.lp test.lp`

Alternatively, you can invoke Draco with `draco -m asp test.lp`.

#### Run APT example

`clingo asp/_apt.lp examples/example_apt.lp --opt-mode=optN --quiet=1 --project -c max_extra_encs=0`

This only prints the relevant data and restricts the extra encodings that are being generated.

## Resources

### Related Repositories

Previous prototypes

* https://github.com/uwdata/vis-csp
* https://github.com/domoritz/vis-constraints

Related software

* https://github.com/uwdata/draco-vis
* https://github.com/vega/compassql
* https://github.com/potassco/clingo

### Guides

* https://github.com/potassco/guide/releases/
