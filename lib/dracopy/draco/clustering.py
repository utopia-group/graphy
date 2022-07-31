import argparse
import collections
import copy
import itertools
import json
import numpy as np
import os
import pandas as pd
import random
import re
import sys

from pprint import pprint

from lib.dracopy.draco.utils import cql_to_asp, data_to_asp


import numpy as np
import sklearn.cluster


#step 1: find similar visualization based on x,y fields
#step 2: compare similarity within the group

def extract_xy_fields(spec):
	encs = spec["encoding"]
	return tuple(sorted(
		[f'{enc["field"]}::{"aggr" if "aggregate" in enc else ""}' if "field" in enc else enc["aggregate"] 
			for ch, enc in spec["encoding"].items() if ch in ["x", "y"]]))

def get_feature(spec):
	"""get features for the spec """
	result = []
	for ch, enc in spec["encoding"].items():
		if "bin" in enc:
			result.append((ch, "bin", enc["bin"]))
		if "aggregate" in enc:
			val = "count" if enc["aggregate"] == "count" else "aggr"
			result.append((ch, "aggregate", True))
		if "field" in enc:
			result.append((ch, "field", enc["field"]))
	return result

def group_based_on_xy_fields(input_vl_specs):
	"""clustering specs based on xy fields."""
	groups = {}
	for vl_spec in input_vl_specs:
		xy_fields = extract_xy_fields(vl_spec)
		if xy_fields not in groups:
			groups[xy_fields] = []
		groups[xy_fields] += [vl_spec]
	return groups

def spec_distance(s1, s2):
	f1, f2 = get_feature(s1), get_feature(s2)
	extra_feat = [x for x in f1 if x not in f2]
	removed_feat = [x for x in f2 if x not in f1]
	return len(extra_feat) + len(removed_feat)

def partition_within_group(specs):

	features = [get_feature(s) for s in specs]
	similarity = -1 * np.array([[spec_distance(s1, s2) for s1 in specs] for s2 in specs])

	# reference algorithm
	# words = ["Georgeann Aquilino", "Carey Mchone", "Ileana Melendrez", "Jalisa Rain", "Brittney Bodily", "Elizabet Buda", "Jetta Valdivia", "Thea Beltrami", "Brandy Ibanez", "Lyndsay Ledger", "Jacqulyn Paredes", "Annita Ehmann", "Olinda Esperanza", "Bess Culver", "Krysta Eis", "Dakota Shimkus", "Loreen Zell", "Hyman Figura", "Akiko Flavell", "Fawn Kleve"]
	# words = np.asarray(words) #So that indexing with a list will work
	# lev_similarity = -1*np.array([[distance.levenshtein(w1,w2) for w1 in words] for w2 in words])

	# affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
	# affprop.fit(lev_similarity)
	# for cluster_id in np.unique(affprop.labels_):
	#     exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
	#     cluster = np.unique(words[np.nonzero(affprop.labels_==cluster_id)])
	#     cluster_str = ", ".join(cluster)
	#     print(" - *%s:* %s" % (exemplar, cluster_str))

	affprop = sklearn.cluster.AffinityPropagation(affinity="precomputed", damping=0.5)
	affprop.fit(similarity)

	buckets = []
	for cluster_id in np.unique(affprop.labels_):
		center = affprop.cluster_centers_indices_[cluster_id]
		neighbors = np.nonzero(affprop.labels_==cluster_id)
		print(neighbors[0])
		buckets.append([specs[int(i)] for i in set(neighbors[0])])

	return buckets

def partition_specs(specs):
	"""partition specs into fine groups so that the distance within group is minimized """
	groups = group_based_on_xy_fields(specs)

	all_buckets = []
	for xy_fields, group in groups.items():
		print(f"## {xy_fields}")
		buckets = partition_within_group(group)
		all_buckets += buckets
	return all_buckets


def build_DAG(specs):
	"""build a DAG to order all specs based on containment relations """
	containment_relations = [(-1, i) for i in range(len(specs))]
	for i, s1 in enumerate(specs):
		for j in range(i + 1, len(specs)):
			s2 = specs[j]
			f1 = get_feature(s1)
			f2 = get_feature(s2)

			# avoid reptition
			if len([x for x in f2 if x not in f1]) == 0:
				# spec i is contained by spec j
				containment_relations.append((i, j))
				continue

			if len([x for x in f1 if x not in f2]) == 0:
				# spec j is contained by spec i
				containment_relations.append((j, i))
	
	to_remove = []
	for rel in containment_relations:
		for i in range(len(specs)):
			if (rel[0], i) in containment_relations and (i, rel[1]) in containment_relations:
				to_remove.append(rel)

	edges = [r for r in containment_relations if r not in to_remove]

	print("# edges:")
	print(edges)

	return edges

if __name__ == '__main__':

	dir_name = "../../vis-viewer/sample-outputs/cars"

	vl_specs = []
	for fname in os.listdir(dir_name):
		if not fname.endswith(".vl.json"):
			continue
		with open(os.path.join(dir_name, fname), "r") as f:
			vl_spec = json.load(f)
			vl_specs.append(vl_spec)

	buckets = partition_specs(vl_specs)

	print(len(vl_specs))
	print(len(buckets))

