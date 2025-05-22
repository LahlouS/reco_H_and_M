import json
import numpy as np
import pandas as pd
import time
from scipy.sparse import load_npz

def open_id_map(filepath):
	with open(filepath) as f:
		id_map = {int(k): v for k, v in json.load(f).items()}
	return id_map

def get_reindex_id(id_map, original_id):
	index_id = [k for k, v in id_map.items() if v == original_id][0]
	return index_id

def recommender(model, reindex_id, interaction_matrix, nreco, filter_already_purshased_items=False):
	start_time = time.time()
	ids, confidence_scores = model.recommend(reindex_id, interaction_matrix[reindex_id], N=nreco, filter_already_liked_items=filter_already_purshased_items)
	end_time = time.time()
	elapsed = end_time - start_time
	return ids, confidence_scores, elapsed

def map_result_to_df(user_id, items_ids, scores, interaction_matrix, item_id_map):
	already_liked = np.in1d(items_ids, interaction_matrix[user_id].indices)
	article_ids = [item_id_map[i] for i in items_ids]
	result = pd.DataFrame({"article_id": article_ids, "score": scores, "already_bought": already_liked})
	return result

def load_interaction_matrix(filepath):
	interaction_matrix = load_npz(filepath)
	return interaction_matrix

