import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz
import json
from implicit.nearest_neighbours import bm25_weight

def get_interaction_matrix(transaction):
	interaction_counts = transaction.groupby(['customer_id', 'article_id']).size().reset_index(name='interaction_count')
	user_encoder = LabelEncoder()
	item_encoder = LabelEncoder()

	interaction_counts['user_idx'] = user_encoder.fit_transform(interaction_counts['customer_id'])
	interaction_counts['item_idx'] = item_encoder.fit_transform(interaction_counts['article_id'])

	# Step 3: Create sparse matrix
	interaction_matrix = csr_matrix((
		interaction_counts['interaction_count'],
		(interaction_counts['user_idx'], interaction_counts['item_idx'])
	))

	user_id_map = dict(zip(interaction_counts['user_idx'], interaction_counts['customer_id']))
	item_id_map = dict(zip(interaction_counts['item_idx'], interaction_counts['article_id']))

	return {
		"interaction_matrix": interaction_matrix,
		"user_id_map": user_id_map,
		"item_id_map": item_id_map
		 }

def remove_neverbought_articles(articles, transactions):
	all_articles = set(articles['article_id'])
	transacted_articles = set(transactions['article_id'])
	missing_articles = all_articles - transacted_articles
	filtered_articles = articles[~articles['article_id'].isin(missing_articles)]
	return filtered_articles

def remove_nevernought_customer(customers, transactions):
	all_customers = set(customers['customer_id'])
	transacting_customers = set(transactions['customer_id'])
	missing_customers = all_customers - transacting_customers
	filtered_customers = customers[~customers['customer_id'].isin(missing_customers)]
	return filtered_customers

def weight_interaction_matrix(interaction_matrix, K1=100, B=0.8):
	# weight the matrix, both to reduce impact of users that have played the same artist thousands of times
	# and to reduce the weight given to popular items
	return bm25_weight(interaction_matrix, K1=K1, B=B)

def save_interaction_matrix(filepath, interaction_matrix):
	save_npz(filepath, interaction_matrix)

def json_save(filepath, id_map):
	id_map_json = {str(k): v for k, v in id_map.items()}
	with open(filepath, "w") as f:
		json.dump(id_map_json, f)