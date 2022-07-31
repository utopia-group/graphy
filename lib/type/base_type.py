from typing import Dict, List, Optional

from lib.utils.misc_utils import printd

"""
Create all the base types here
"""


class BaseType:
    def __init__(self, name: str):
        self.name = name

    def update(self, **args):
        raise NotImplementedError

    def duplicate(self) -> 'BaseType':
        raise NotImplementedError

    def get_vegalite_name(self) -> str:
        return ''

    def __repr__(self):
        return self.name


class ConstColType(BaseType):
    def __init__(self, name: str = 'ConstCol', col: Optional[str] = None):
        super(ConstColType, self).__init__(name)
        self.col = col


class ConstType(BaseType):
    def __init__(self, name: str = 'Const'):
        super().__init__(name)


class PlotType(BaseType):
    def __init__(self, name: str = 'Plot'):
        super().__init__(name)


class BarPlot(PlotType):
    def __init__(self, name: str = 'BarPlot'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'bar'


class Histogram(PlotType):
    def __init__(self, name: str = 'Histogram'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'bar'


class ScatterPlot(PlotType):
    def __init__(self, name: str = 'ScatterPlot'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'point'


class LinePlot(PlotType):
    def __init__(self, name: str = 'LinePlot'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'line'


class AreaPlot(PlotType):
    def __init__(self, name: str = 'AreaPlot'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'area'


class CellType(BaseType):
    def __init__(self, name: str = 'Cell'):
        super().__init__(name)


class QualitativeType(CellType):
    def __init__(self, name: str = 'Qualitative'):
        super().__init__(name=name)

    def get_vegalite_name(self) -> str:
        return 'nominal'


class QuantitativeType(CellType):
    def __init__(self, name: str = 'Quantitative'):
        super().__init__(name=name)

    def get_vegalite_name(self) -> str:
        return 'quantitative'


class DiscreteType(QuantitativeType):
    def __init__(self, name: str = 'Discrete'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'quantitative'


class ContinuousType(QuantitativeType):
    def __init__(self, name: str = 'Continuous'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'quantitative'


class OrdinalType(QualitativeType):
    def __init__(self, name: str = 'Ordinal'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'ordinal'


class NominalType(QualitativeType):
    def __init__(self, name: str = 'Nominal'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'nominal'


class TemporalType(QualitativeType):
    def __init__(self, name: str = 'Temporal'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'ordinal'


class AggregateType(QuantitativeType):
    def __init__(self, name: str = 'Aggregate'):
        super().__init__(name)

    def get_vegalite_name(self) -> str:
        return 'quantitative'


class NullType(BaseType):
    def __init__(self, name: str = 'Null'):
        super().__init__(name)


class NullOpType(BaseType):
    def __init__(self, name: str = 'NullOp'):
        super().__init__(name)


class ListType(BaseType):
    def __init__(self, T: CellType, name: str = 'List'):
        super().__init__(name)
        self.T: CellType = T

    def __repr__(self):
        return 'List[{}]'.format(repr(self.T))


class AlphaType(BaseType):
    def __init__(self, name: str = 'Table'):
        super().__init__(name)


class TableType(BaseType):
    """
    Table Type: map a set of column names to their types (all base types)
    """

    def __init__(self, fields: Optional[List[str]] = None, name: str = 'Table'):
        super().__init__(name)
        self.fields = fields
        if fields is None:
            self.__record: Dict[str, ListType] = {}
        else:
            self.__record: Dict[str, ListType] = dict(
                (field, ListType(CellType())) for field in fields)

    def duplicate(self) -> 'TableType':
        new_type = TableType()
        new_type.fields = self.fields
        new_type.__record = self.__record.copy()

        return new_type

    def replace(self, prev_col: str, new_col: str):
        """
        replace the old key with new key
        """
        self.__record[new_col] = self.__record.pop(prev_col)

    def add(self, col: str, _type: CellType):
        self.__record[col] = ListType(_type)

    def update_record(self, col: str, new_type: ListType):
        # print("update_record", col, self.__record[col], new_type)
        if new_type.T == self.__record[col].T:
            pass
        elif isinstance(new_type.T, self.__record[col].T.__class__):
            self.__record[col] = new_type
        elif isinstance(self.__record[col].T, new_type.T.__class__):
            # self.__record[col] = new_type
            pass
        else:
            # print("type error")
            raise TypeError(
                'Cannot update type of {} to {}'.format(self.__record[col].T, new_type.T))

    def update(self, col: str, new_type: ListType, check_subtype=False):
        """
        update the record
        make sure new_type has to be a subtype of previous type
        """
        # print(self)
        if check_subtype and new_type.T == self.__record[col].T:
            pass
        elif (not check_subtype) or isinstance(new_type.T, self.__record[col].T.__class__):
            self.__record[col] = new_type
        else:
            printd("TypeError: Failed to update table type for col {}({}) with new type {}".format(
                col, self.__record[col], new_type))
            raise TypeError

    def get_record(self) -> Dict[str, ListType]:
        return self.__record

    def __repr__(self):
        return str(self.__record)
