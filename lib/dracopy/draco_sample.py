import draco
spec = ['num_rows(303).', 'fieldtype("Model",string).', 'cardinality("Model", 303).', 'fieldtype("MPG",number).', 'cardinality("MPG", 111).', 'fieldtype("Cylinders",number).', 'cardinality("Cylinders", 5).', 'fieldtype("Displacement",number).', 'cardinality("Displacement", 78).', 'fieldtype("Horsepower",number).', 'cardinality("Horsepower", 89).', 'fieldtype("Weight",number).', 'cardinality("Weight", 274).', 'fieldtype("Acceleration",number).', 'cardinality("Acceleration", 91).', 'fieldtype("Year",number).', 'cardinality("Year", 13).', 'fieldtype("Origin",string).', 'cardinality("Origin", 3).', 'encoding(e0).', ':- not field(e0,"Horsepower").', ':- not bin(e0, _).']

spec = ['num_rows(303).', 'fieldtype("Horsepower",number).', 'cardinality("Horsepower", 89).', 'encoding(e0).', ':- not field(e0,"Horsepower").', ':- not bin(e0, _).']
files = ['_saket2018.lp']
additional_options=['--opt-mode=optN']
res = draco.run(spec)
res.as_vl()