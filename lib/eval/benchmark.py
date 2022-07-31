import itertools
from collections import defaultdict

import pandas as pd

from typing import Dict, Tuple, List, Union, Optional, FrozenSet

from pandas import DataFrame

from lib.trinity.table import Table
from lib.type.base_type import TableType
from lib.type.predicate import Prov, RelationPred, Card, Constant, Predicate
from lib.type.ref_type import BaseRefType
from lib.type.type_system import get_base_type, create_cnf_formula_instance, create_list_type
from lib.type.formula import create_term
from lib.utils.csv_utils import read_csv_to_dict


class Dataset:
    """
    Wrapper for the dataset, include additional analysis information
    """

    def __init__(self, dataname, benchmark_set, data_dir='eval/data', analyze_data_nl2sql=False, analyze_data_syn=False):

        self.data_dir: str = data_dir
        self.dataname: str = dataname
        self.benchmark_set: str = benchmark_set

        # the movie dataset has some minor difference between the chi21 and the nl4dv version
        self.data_path: str = "{0}/{1}_{2}/{1}.csv".format(data_dir, dataname, benchmark_set) \
            if dataname.lower() == 'movies' else \
            "{0}/{1}/{1}.csv".format(data_dir, dataname)

        self.data, self.data_df = self.read_data()

        self.colnames = list(self.data[0].keys())

        # the following is used in the nl2sql parser for predicate parsing (which we are actually not using)
        self.field_to_type_mapping = {}
        self.field_to_val_mapping = {}
        if analyze_data_nl2sql:
            self.analyze_data_nl2sql()

        # the following are relevant data information: type-general, type-detailed (maybe using NER), cardinality
        # I decide to hand-code for now
        self.all_constraints: Dict[str, List] = {}
        self.datatype_properties: Dict[str, Dict[str, Union[str, int]]] = {}
        self.cardinality_properties: Dict[FrozenSet, int] = {}
        if analyze_data_syn:
            self.analyze_data_syn()

    def read_data(self) -> Tuple[List[Dict], DataFrame]:
        return read_csv_to_dict(self.data_path), pd.read_csv(self.data_path)

    def get_df(self):
        return self.data_df

    def analyze_data_syn(self):

        if self.dataname.lower() == 'cars':
            self.datatype_properties = {
                'Model': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'MPG': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Displacement': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Cylinders': {'type': 'Discrete', 'type-f': 'number', 'type-g': 'number'},
                'Acceleration': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Horsepower': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Weight': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Year': {'type': 'Ordinal', 'type-f': 'date', 'type-g': 'number'},
                'Origin': {'type': 'Nominal', 'type-f': 'loc', 'type-g': 'string'},
            }
        elif self.dataname.lower() == 'movies':
            self.datatype_properties = {
                'Title': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Worldwide_Gross': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Production_Budget': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Release_Year': {'type': 'Temporal', 'type-f': 'date', 'type-g': 'number'},
                'Content_Rating': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Running_Time': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Major_Genre': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Creative_Type': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Rotten_Tomatoes_Rating': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'IMDB_Rating': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
            }
        elif self.dataname.lower() == 'superstore':
            self.datatype_properties = {
                'Days_to_Ship_Actual': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Sales_Forecast': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Ship_Status': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Days_to_Ship_Scheduled': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Sales_per_Customer': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Profit_Ratio': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Category': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'City': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Country': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Customer_Name': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Days_to_Ship_Actual_(bin)': {'type': 'Discrete', 'type-f': 'number', 'type-g': 'number'},
                'Discount': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Number_of_Records': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Order_Date': {'type': 'Temporal', 'type-f': 'date', 'type-g': 'temporal'},
                'Order_ID': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Postal_Code': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Product_Name': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Profit': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Quantity': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Region': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Sales_(bin)': {'type': 'Discrete', 'type-f': 'number', 'type-g': 'number'},
                'Sales': {'type': 'Continuous', 'type-f': 'number', 'type-g': 'number'},
                'Segment': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Ship_Date': {'type': 'Ordinal', 'type-f': 'date', 'type-g': 'temporal'},
                'Ship_Mode': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'State': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
                'Sub-Category': {'type': 'Nominal', 'type-f': 'string', 'type-g': 'string'},
            }

        else:
            raise NotImplementedError('Only support dataset in the benchmarks so far.')

        # create a dict for cardinality analysis only
        self.cardinality_properties = {}
        for (columnName, columnData) in self.data_df.iteritems():
            self.cardinality_properties[frozenset([columnName])] = columnData.nunique()

        # every two combination cardinality
        for subtable in itertools.combinations(self.datatype_properties.keys(), 2):
            self.cardinality_properties[frozenset(subtable)] = len(self.data_df[list(subtable)].drop_duplicates())

        # every three combination cardinality
        for subtable in itertools.combinations(self.datatype_properties.keys(), 3):
            self.cardinality_properties[frozenset(subtable)] = len(self.data_df[list(subtable)].drop_duplicates())

        # TODO: add min max, and more fine-grained cardinality analysis here. I am skipping a couple of them.

    def analyze_data_nl2sql(self):
        datatype_path = "{0}/{1}_{2}/{1}.types".format(self.data_dir, self.dataname, self.benchmark_set) \
            if self.dataname.lower() == "movies" \
            else "{0}/{1}/{1}.types".format(self.data_dir, self.dataname)
        with open(datatype_path) as f:
            types = [line.rstrip() for line in f]
        # print("analyze_data_nl2sql: ", types)
        assert len(types) == len(self.colnames)
        for i in range(len(types)):
            self.field_to_type_mapping[self.colnames[i]] = types[i]

        for colname in self.colnames:
            self.field_to_val_mapping[colname] = set([r[colname].lower() for r in self.data])

    """
    get certain types of columns (for assign x, y axis col and hallucination purpose)
    """

    def get_categorical(self) -> List:
        fields = []
        for field, _property in self.datatype_properties.items():
            if _property['type'] == 'Ordinal' or _property['type'] == 'Discrete' or _property['type'] == 'Nominal':
                if self.cardinality_properties[frozenset([field])] <= 5:
                    fields.append(field)

        return fields

    """
    draco related functions (get constraints)
    """

    def get_data_path_constraint(self):
        return 'data("{}")'.format(self.data_path)

    def get_constraints(self, fields=None) -> Dict:

        if fields is None:
            if len(self.all_constraints) == 0:
                self.all_constraints = dict([(field, ['fieldtype("{}",{}).'.format(field, self.datatype_properties[field]['type-g']),
                                                      'cardinality("{}",{}).'.format(field, self.cardinality_properties[frozenset([field])])]) for field in self.colnames])
            return self.all_constraints

        constraints = dict([(field, ['fieldtype("{}",{}).'.format(field, self.datatype_properties[field]['type-g']),
                                     'cardinality("{}",{}).'.format(field, self.cardinality_properties[frozenset([field])])]) for field in fields])
        return constraints


class InputDataset(Dataset):
    """
    More specific class with type information for the input Dataset, create for the synthesizer use only
    """

    def __init__(self, dataname, benchmark_set, data_dir='eval/data'):
        dataname = dataname.lower()
        super().__init__(dataname, benchmark_set, data_dir, analyze_data_nl2sql=True, analyze_data_syn=True)

        self.input_type: Optional[BaseRefType] = None
        self.data_constraint_pool: Dict[FrozenSet, Tuple[RelationPred, bool]] = {}
        self.syntax_constraint_pool: Dict[str, List[List[Tuple[Predicate, bool]]]] = defaultdict(list)

        print("Finished init input dataset")

    def init_ref_type(self):
        """
        according to the dataset, generate a bunch of constraint pool that we can possibly use in the future
        """

        # let's create a bunch of constraints
        cols: FrozenSet
        cardinality: int
        for cols, cardinality in self.cardinality_properties.items():
            self.data_constraint_pool[cols] = (RelationPred('eq', create_term(Card, list(cols)), create_term(Constant, cardinality)), False)

        col: str
        for col in self.colnames:
            for beta in ['mean', 'sum', 'count']:
                self.syntax_constraint_pool[col].append([(Prov(col, beta), True)])

    def update_input_type(self, colnames: Optional[List[str]], all_columns: bool = False):
        """
        we update the input type according to the output of the nlp model
        so our input table type only includes information we care
        """
        colnames = self.colnames if all_columns else colnames

        input_base_type = TableType(colnames)
        constraint = [[(RelationPred('eq', create_term(Card, 'T'), create_term(Constant, len(self.data_df[colnames]))), False)]]

        for col in colnames:
            input_base_type.update(col, create_list_type(self.datatype_properties[col]['type']))
            constraint.extend(self.syntax_constraint_pool[col])
            constraint.append([self.data_constraint_pool[frozenset([col])]])

        for i in range(2, len(colnames)):
            for subtable in itertools.combinations(colnames, i):
                if len(subtable) > 3:
                    continue
                constraint.append([self.data_constraint_pool[frozenset(subtable)]])


        self.input_type = BaseRefType(input_base_type, create_cnf_formula_instance(constraint))

    def update_input_type_without_datatype_or_card(self, colnames: Optional[List[str]], all_columns: bool = False):
        """
        Update input type to include column names and syntactic constraints,
        but no information on datatype or cardinality of columns
        """
        colnames = self.colnames if all_columns else colnames

        input_base_type = TableType(colnames)
        constraint = []

        for col in colnames:
            input_base_type.update(col, create_list_type('Cell'))
            constraint.extend(self.syntax_constraint_pool[col])

        self.input_type = BaseRefType(input_base_type, create_cnf_formula_instance(constraint))

    def update_input_type_without_card(self, colnames: Optional[List[str]], all_columns: bool = False):
        """
        Update input type to include column names and syntactic constraints,
        but no information on cardinality of columns
        """
        colnames = self.colnames if all_columns else colnames

        input_base_type = TableType(colnames)
        constraint = []

        for col in colnames:
            input_base_type.update(col, create_list_type(self.datatype_properties[col]['type']))
            constraint.extend(self.syntax_constraint_pool[col])

        self.input_type = BaseRefType(input_base_type, create_cnf_formula_instance(constraint))

    def update_input_type_without_datatype(self, colnames: Optional[List[str]], all_columns: bool = False):
        """
        Update input type to include column names and syntactic constraints,
        but no information on datatype or cardinality of columns
        """
        colnames = self.colnames if all_columns else colnames

        input_base_type = TableType(colnames)
        constraint = [[(RelationPred('eq', create_term(Card, 'T'), create_term(Constant, len(self.data_df[colnames]))), False)]]

        for col in colnames:
            input_base_type.update(col, create_list_type('Cell'))
            constraint.extend(self.syntax_constraint_pool[col])
            constraint.append([self.data_constraint_pool[frozenset([col])]])

        for i in range(2, len(colnames)):
            for subtable in itertools.combinations(colnames, i):
                if len(subtable) > 3:
                    continue
                constraint.append([self.data_constraint_pool[frozenset(subtable)]])

        self.input_type = BaseRefType(input_base_type, create_cnf_formula_instance(constraint))


class Benchmark:
    """
    Wrapper for a benchmark instance
    """

    def __init__(self, dataname=None, bname=None, nl=None, gtname=None, benchmark_set=None, data=None):
        self.dataname = dataname
        self.bname = bname
        self.nl = nl
        self.gtname = gtname
        self.benchmark_set = benchmark_set

        self.data: Union[Dataset, InputDataset, Table] = data
        self.spec = None
        self.fields = None

    def get_id(self):
        return "{}-{}".format(self.dataname, self.bname)

    def get_neural_format(self) -> Dict:
        return {'id': self.bname, 'data': self.dataname, 'query': self.nl, 'query-fixed': '', 'gtname': self.gtname}
