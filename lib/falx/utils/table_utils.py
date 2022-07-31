import json
import pandas as pd
import re

from pandas.core.frame import DataFrame

def infer_dtype(values):
	return pd.api.types.infer_dtype(values, skipna=True)

def filter_table(table, pred):
	"""convert js expression to python expression and then run eval """
	pred = pred.replace("&&", "and").replace("||", "or")
	pred = " ".join([v if "datum" not in v else "[\"".join(v.split(".")) + "\"]" for v in pred.split()])
	res = [datum for datum in table if eval(pred)]
	return res

def apply_transform(table, transform):
	"""apply filtering operation in the transform to the table"""

	if transform is None:
		return table

	df_original = load_and_clean_dataframe(pd.DataFrame.from_dict(table))

	if type(table) == list:
		df = load_and_clean_dataframe(pd.DataFrame.from_dict(table), special_datetime=True)
	elif type(table) == DataFrame:
		df = load_and_clean_dataframe(table, special_datetime=True)
	else:
		print("{} table type not supported".format(type(table)))
		raise Exception

	# for col in df:
	# 	dtype, new_col_values = clean_column_dtype(df[col])
	# 	print("{}: {}".format(col, dtype))

	preds = []
	for t in transform:
		pred = t["filter"]
		# check negation
		negation = "not" in pred
		if negation:
			pred = t["filter"]["not"]

		field = pred["field"]
		for key, value in pred.items():
			if key == "field":
				pass
			elif key == "equal":
				if negation:
					preds.append("(df[{}] != {})".format(repr(field), repr(value)))
				else:
					preds.append("(df[{}] == {})".format(repr(field), repr(value)))
			elif key == "lt":
				preds.append("(df[{}] < {})".format(repr(field), repr(value)))
			elif key == "lte":
				preds.append("(df[{}] <= {})".format(repr(field), repr(value)))
			elif key == "gt":
				preds.append("(df[{}] > {})".format(repr(field), repr(value)))
			elif key == "gte":
				preds.append("(df[{}] >= {})".format(repr(field), repr(value)))
			elif key == "oneOf":
				preds.append("({})".format("|".join(["(df[{}] == {})".format(repr(field), repr(v)) for v in value])))
			elif key == "range":
				preds.append("((df[{0}] >= {1}) & (df[{0}] <= {2}))".format(repr(field), repr(value[0]), repr(value[1])))
			elif key == "valid":
				preds.append("((df[{0}] != null) & (df[{0}] != np.nan))".format(repr(field)))
			else:
				print("operation {} not supported".format(key))
				raise Exception
	
	preds_str = "&".join(preds)
	print("preds: ", preds_str)
	mask = eval(preds_str)
	filtered_data = df_original[mask]

	# print(filtered_data.to_dict(orient="records"))
	
	if type(table) == list:
		return filtered_data.to_dict(orient="records")
	else:
		return filtered_data

def table_subset_eq(table1, table2):
	"""check whether table1 is subsumed by table2 """
	if len(table1) == 0: return True
	if len(table2) == 0: return False

	schema1 = tuple(sorted(table1[0].keys()))
	schema2 = tuple(sorted(table2[0].keys()))
	if schema1 != schema2: return False

	frozen_table1 = [tuple([t[key] for key in schema1]) for t in table1]
	frozen_table2 = [tuple([t[key] for key in schema1]) for t in table2]

	for t in frozen_table1:
		cnt1 = len([r for r in frozen_table1 if r == t])
		cnt2 = len([r for r in frozen_table2 if r == t])
		if cnt2 < cnt1:
			return False
	return True

def clean_column_dtype(column_values, colname=None, special_datetime=False):
	dtype = pd.api.types.infer_dtype(column_values, skipna=True)

	# print("dtype:", dtype)

	if special_datetime and colname is not None and "year" in colname.lower() and dtype == "integer":
		try:
			values = pd.to_datetime(column_values, format="%Y")
			dtype = pd.api.types.infer_dtype(values, skipna=True)
			# print(values)
			return dtype, values
		except:
			return dtype, column_values

	if dtype != "string":
		return dtype, column_values

	def try_infer_string_type(values):
		"""try to infer datatype from values """
		dtype = pd.api.types.infer_dtype(values, skipna=True)
		ty_check_functions = [
			lambda l: pd.to_numeric(l),
			lambda l: pd.to_datetime(l, infer_datetime_format=True)
		]
		for ty_func in ty_check_functions:
			try:
				values = ty_func(values)
				dtype = pd.api.types.infer_dtype(values, skipna=True)
			except Exception as e:
				# print(e)
				pass
			if dtype != "stirng":
				break
		return dtype, values

	def to_time(l):
		return l[0] * 60 + l[1]

	convert_functions = {
		"id": (lambda l: True, lambda l: l),
		"percentage": (lambda l: all(["%" in x for x in l]), 
					   lambda l: [x.replace("%", "").replace(" ", "") if x.strip() not in [""] else "" for x in l]),
		"currency": (lambda l: True, lambda l: [x.replace("$", "").replace(",", "") for x in l]),
		"cleaning_missing_number": (lambda l: True, lambda l: [x if x.strip() not in [""] else "" for x in l]),
		"cleaning_time_value": (lambda l: True, lambda l: [to_time([int(y) for y in x.split(":")]) for x in l]),
	}

	for key in convert_functions:
		if convert_functions[key][0](column_values):
			try:
				converted_values = convert_functions[key][1](column_values)
			except:
				continue
			dtype, values = try_infer_string_type(converted_values)
		if dtype != "string": 
			if key == "percentage":
				values = values / 100.
			break
	return dtype, values

def load_and_clean_dataframe(df, special_datetime=False):
	"""infer type of each column and then update column value
	Args:
		df: input dataframe we want to clean
	Returns:
		clean data frame
	"""
	for col in df:
		# print("col:", col)
		dtype, new_col_values = clean_column_dtype(df[col], colname=col, special_datetime=special_datetime)
		df[col] = new_col_values
	return df

def load_and_clean_table(input_data, return_as_df=False):
	"""load and clean table where the input format is a table record """
	try:
		df = load_and_clean_dataframe(pd.DataFrame.from_dict(input_data))
		if return_as_df:
			return df
		else:
			return df.to_dict(orient="records")
	except:
		print("# [warning] error cleaning table, return without cleaning")
		return input_data