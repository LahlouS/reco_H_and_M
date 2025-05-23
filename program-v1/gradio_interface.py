import gradio as gr
import random
import pandas as pd
import json
import os
from ALS_predict_lib import open_id_map, get_reindex_id, load_interaction_matrix, recommender, map_result_to_df
from implicit.als import AlternatingLeastSquares



CUSTOMERS_IDS_MAP_PATH = "/home/slahlou/Documents/recoGnomon/dataset/collaborative-filtering-data/user_id_map.json"
ITEMS_IDS_MAP_PATH = "/home/slahlou/Documents/recoGnomon/dataset/collaborative-filtering-data/item_id_map.json"
INTERACTION_MATRIX_PATH = "/home/slahlou/Documents/recoGnomon/dataset/collaborative-filtering-data/interaction_matrix.npz"
MODEL_WEIGHTS_PATH = "/home/slahlou/Documents/recoGnomon/program-v1/weights/model_weights_20250522_234808.npz"

TRANSACTION_DATASET_PATH = "/home/slahlou/Documents/recoGnomon/dataset/datas/transactions_train.csv"
ARTICLES_DATASET_PATH = "/home/slahlou/Documents/recoGnomon/dataset/datas/articles.csv"


def fetch_available_user_ids():
	# customer_id_map = open_id_map(CUSTOMERS_IDS_MAP_PATH)
	customer_id_map = {
			3: "00005ca1c9ed5f5146b52ac8639a40ca9d57aeff4d1bd2c5feb1ca5dff07c43e",
			4: "00006413d8573cd20ed7128e53b7b13819fe5cfc2d801fe7fc0f26dd8d65a85a",
			5: "000064249685c11552da43ef22a5030f35a147f723d5b02ddd9fd22452b1f5a6",
			6: "0000757967448a6cb83efb3ea7a3fb9d418ac7adf2379d8cd0c725276a467a2a",
			7: "00007d2de826758b65a93dd24ce629ed66842531df6699338c5570910a014cc2",
			8: "00007e8d4e54114b5b2a9b51586325a8d0fa74ea23ef77334eaec4ffccd7ebcc",
			9: "00008469a21b50b3d147c97135e25b4201a8c58997f78782a0cc706645e14493",
			10: "0000945f66de1a11d9447609b8b41b1bc987ba185a5496ae8831e8493afa24ff",
			11: "000097d91384a0c14893c09ed047a963c4fc6a5c021044eec603b323e8c82d1d",
			12: "00009c2aeae8761f738e4f937d9be6b49861a66339c2b1c3b1cc6e322729a370",
			13: "00009d946eec3ea54add5ba56d5210ea898def4b46c68570cf0096d962cacc75",
			14: "0000ae1bbb25e04bdc7e35f718e852adfb3fbb72ef38b3fa01ce4272a6326730"
		}
	return customer_id_map


WEIGHTS = {
	"Base case": 1.0,
	"AlternativeLeastSquare": 1.0,
	"Co-purshased score": 1.0,
	"logistic_regression":1.0
}

def parse_dict_input(input_text):
	"""Safely parses a user input string into a dictionary."""
	try:
		return json.loads(input_text)
	except json.JSONDecodeError:
		return None

def format_elapsed_time_html(elapsed_time):
    """Format elapsed time as HTML"""
    html = "<div style='font-family: Arial, sans-serif; max-width: 600px; margin-top: 20px;'>"
    html += f"""
    <div style='border: 1px solid #2ecc71; padding: 15px; border-radius: 8px; background-color: #d5f4e6; border-left: 4px solid #27ae60;'>
        <div style='display: flex; align-items: center; justify-content: center;'>
            <span style='font-size: 24px; margin-right: 10px;'>⏱️</span>
            <div>
                <h4 style='color: #27ae60; margin: 0 0 5px 0; font-size: 16px; font-weight: bold;'>Inference Time</h4>
                <p style='color: #2c3e50; margin: 0; font-size: 18px; font-weight: bold;'>{elapsed_time*1000:.4f} ms</p>
                <small style='color: #7f8c8d; font-size: 12px;'>Time taken to generate recommendations</small>
            </div>
        </div>
    </div>
    """
    html += "</div>"
    return html

def format_purchased_products_html(purchased_products):
	"""Format user's previously purchased products as HTML"""
	html = "<div style='font-family: Arial, sans-serif; max-width: 600px;'>"
	html += "<h3 style='color: #2c3e50; margin-bottom: 15px; border-bottom: 2px solid #3498db; padding-bottom: 5px;'>Previously Purchased Products</h3>"
	
	if not purchased_products:
		html += "<p style='color: #7f8c8d; font-style: italic;'>No previous purchases found.</p>"
	else:
		for product in purchased_products:
			# Create product description from available fields
			description_parts = []
			if product.get('detail_desc') and product['detail_desc'] != 'No description available':
				description_parts.append(product['detail_desc'])
			if product.get('colour_group'):
				description_parts.append(f"Color: {product['colour_group']}")
			if product.get('graphical_appearance'):
				description_parts.append(f"Style: {product['graphical_appearance']}")
			
			description = " • ".join(description_parts) if description_parts else "No description available"
			
			html += f"""
			<div style='border: 1px solid #ddd; padding: 12px; margin-bottom: 10px; border-radius: 6px; background-color: #f8f9fa; border-left: 4px solid #3498db;'>
				<h4 style='color: #2c3e50; margin: 0 0 5px 0; font-size: 16px;'>{product.get('prod_name', 'Unknown Product')}</h4>
				<div style='margin-bottom: 8px;'>
					<span style='background-color: #3498db; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 5px;'>{product.get('product_type', 'Unknown Type')}</span>
					{f"<span style='background-color: #95a5a6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 5px;'>{product.get('department', '')}</span>" if product.get('department') else ""}
					{f"<span style='background-color: #7f8c8d; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>{product.get('section', '')}</span>" if product.get('section') else ""}
				</div>
				<p style='color: #555; margin: 0 0 8px 0; font-size: 13px; line-height: 1.4;'>{description}</p>
				<div style='display: flex; justify-content: space-between; align-items: center;'>
					<small style='color: #7f8c8d; font-size: 11px;'>Article ID: {product.get('product_id', 'N/A')}</small>
					{f"<small style='color: #8e44ad; font-size: 11px;'>{product.get('garment_group', '')}</small>" if product.get('garment_group') else ""}
				</div>
			</div>
			"""
	
	html += "</div>"
	return html

def format_recommended_products_html(recommended_products):
	"""Format recommended products as HTML"""
	html = "<div style='font-family: Arial, sans-serif; max-width: 600px;'>"
	html += "<h3 style='color: #2c3e50; margin-bottom: 15px; border-bottom: 2px solid #e74c3c; padding-bottom: 5px;'>Recommended Products</h3>"
	
	if not recommended_products:
		html += "<p style='color: #7f8c8d; font-style: italic;'>No recommendations available.</p>"
	else:
		for i, product in enumerate(recommended_products, 1):
			# Create product description from available fields
			description_parts = []
			if product.get('detail_desc') and product['detail_desc'] != 'No description available':
				description_parts.append(product['detail_desc'])
			if product.get('colour_group'):
				description_parts.append(f"Color: {product['colour_group']}")
			if product.get('graphical_appearance'):
				description_parts.append(f"Style: {product['graphical_appearance']}")
			
			description = " • ".join(description_parts) if description_parts else "No description available"
			
			# Check if already bought
			already_bought_badge = ""
			if product.get('already_bought', False):
				already_bought_badge = "<span style='background-color: #f39c12; color: white; padding: 2px 6px; border-radius: 4px; font-size: 10px; margin-left: 10px;'>REPURCHASE</span>"
			
			html += f"""
			<div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 15px; border-radius: 8px; background-color: #fff; border-left: 4px solid #e74c3c; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
				<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>
					<h4 style='color: #2c3e50; margin: 0; font-size: 16px;'>{product.get('prod_name', 'Unknown Product')}{already_bought_badge}</h4>
					<span style='background-color: #e74c3c; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;'>#{i}</span>
				</div>
				<div style='margin-bottom: 10px;'>
					<span style='background-color: #e74c3c; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 5px;'>{product.get('product_type', 'Unknown Type')}</span>
					{f"<span style='background-color: #95a5a6; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px; margin-right: 5px;'>{product.get('department', '')}</span>" if product.get('department') else ""}
					{f"<span style='background-color: #7f8c8d; color: white; padding: 2px 6px; border-radius: 4px; font-size: 11px;'>{product.get('section', '')}</span>" if product.get('section') else ""}
				</div>
				<p style='color: #555; margin: 0 0 12px 0; font-size: 13px; line-height: 1.5;'>{description}</p>
				<div style='display: flex; justify-content: space-between; align-items: center;'>
					<div style='display: flex; align-items: center; gap: 15px;'>
						<small style='color: #7f8c8d; font-size: 11px;'>Article ID: {product.get('product_id', 'N/A')}</small>
						{f"<small style='color: #8e44ad; font-size: 11px;'>{product.get('garment_group', '')}</small>" if product.get('garment_group') else ""}
					</div>
					<div style='display: flex; align-items: center; gap: 10px;'>
						{f"<small style='color: #27ae60; font-weight: bold; font-size: 12px;'>Score: {product.get('recommendation_score', 0):.3f}</small>" if product.get('recommendation_score') is not None else ""}
					</div>
				</div>
			</div>
			"""
	
	html += "</div>"
	return html

def format_user_ids(user_dict):
	"""Format available user IDs for display"""
	return "\n".join([f"{idx}: {user_id}" for idx, user_id in user_dict.items()])

def get_user_purchased_products(user_id):
	"""Fetch user's previously purchased products from database"""
	transaction_df = pd.read_csv(TRANSACTION_DATASET_PATH)
	articles_df = pd.read_csv(ARTICLES_DATASET_PATH)
	customer_transaction = transaction_df[transaction_df['customer_id'] == user_id]

	col_to_keep = [
		"article_id",
		"prod_name",
		"product_type_name",
		"product_group_name",
		"graphical_appearance_name",
		"colour_group_name",
		"department_name",
		"index_name",
		"index_group_name",
		"section_name",
		"garment_group_name",
		"detail_desc"
	]

	customer_transaction = pd.merge(customer_transaction, 
									articles_df[col_to_keep], 
									on='article_id', 
									how="left")
	
	purchased_products = []
	for _, row in customer_transaction.iterrows():
		product_dict = {
			"product_id": row['article_id'],
			"prod_name": row.get('prod_name', 'Unknown Product'),
			"product_type": row.get('product_type_name', 'Unknown Type'),
			"product_group": row.get('product_group_name', ''),
			"department": row.get('department_name', ''),
			"section": row.get('section_name', ''),
			"garment_group": row.get('garment_group_name', ''),
			"colour_group": row.get('colour_group_name', ''),
			"graphical_appearance": row.get('graphical_appearance_name', ''),
			"index_name": row.get('index_name', ''),
			"index_group": row.get('index_group_name', ''),
			"detail_desc": row.get('detail_desc', 'No description available')
		}
		purchased_products.append(product_dict)
	
	return purchased_products

def run_recommendation_engine(user_id, weights):
	"""Run the recommendation engine for a specific user"""
	articles_df = pd.read_csv(ARTICLES_DATASET_PATH)
	weights = weights if weights is not None else WEIGHTS
	
	customers_id_map = open_id_map(CUSTOMERS_IDS_MAP_PATH)
	items_id_map = open_id_map(ITEMS_IDS_MAP_PATH)

	interaction_matrix = load_interaction_matrix(INTERACTION_MATRIX_PATH)

	model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0).load(MODEL_WEIGHTS_PATH)

	index_id = get_reindex_id(customers_id_map, original_id=user_id)

	ids, confidence_scores, elapsed = recommender(model, 
													index_id, 
													interaction_matrix, 
													nreco=5, 
													filter_already_purshased_items=True)

	result_df = map_result_to_df(index_id, ids, confidence_scores, interaction_matrix, items_id_map)

	col_to_keep = [
		"article_id",
		"prod_name",
		"product_type_name",
		"product_group_name",
		"graphical_appearance_name",
		"colour_group_name",
		"department_name",
		"index_name",
		"index_group_name",
		"section_name",
		"garment_group_name",
		"detail_desc"
	]

	result_df = pd.merge(result_df, 
							articles_df[col_to_keep],
							on='article_id', 
							how="left")

	# Convert dataframe to list of dictionaries
	recommendations = []
	for _, row in result_df.iterrows():
		product_dict = {
			"product_id": row['article_id'],
			"prod_name": row.get('prod_name', 'Unknown Product'),
			"product_type": row.get('product_type_name', 'Unknown Type'),
			"product_group": row.get('product_group_name', ''),
			"department": row.get('department_name', ''),
			"section": row.get('section_name', ''),
			"garment_group": row.get('garment_group_name', ''),
			"colour_group": row.get('colour_group_name', ''),
			"graphical_appearance": row.get('graphical_appearance_name', ''),
			"index_name": row.get('index_name', ''),
			"index_group": row.get('index_group_name', ''),
			"detail_desc": row.get('detail_desc', 'No description available'),
			"recommendation_score": row.get('score', 0.0),
			"already_bought": row.get('already_bought', False)
		}
		recommendations.append(product_dict)
	
	return recommendations, elapsed

def get_user_recommendations(user_id, weights):
	"""Main function to get user data and recommendations"""
	if not user_id:
		return "", ""
	
	# Parse weights
	weights_dict = parse_dict_input(weights)
	
	# Get user's purchased products
	purchased_products = get_user_purchased_products(str(user_id))
	formatted_purchased = format_purchased_products_html(purchased_products)
	
	# Get recommendations
	recommended_products, elapsed = run_recommendation_engine(str(user_id), weights_dict)
	formatted_recommendations = format_recommended_products_html(recommended_products)
	formatted_elapsed = format_elapsed_time_html(elapsed)
	
	return formatted_purchased, formatted_recommendations, formatted_elapsed

# Create Gradio interface
with gr.Blocks(title="User Product Recommendation System") as demo:
	gr.Markdown("# User Product Recommendation System")
	gr.Markdown("Enter a User ID and customize weights to get personalized product recommendations")
	
	with gr.Row():
		with gr.Column(scale=1):  # Left Side
			user_id = gr.Textbox(label="User ID", value='84eba3a874e7b5660ee2803fe64ce82954a2beb56e39ed9a561e01889b9b82b8')
			weights = gr.Textbox(
				label="Recommendation Weights (JSON format)", 
				value=json.dumps(WEIGHTS, indent=2),
				lines=8
			)
			
			# Available user IDs (you would populate this from your database)
			try:
				users = format_user_ids(fetch_available_user_ids())
			except:
				users = "Database connection not available - using demo mode"
			
			available_users = gr.Textbox(
				label="Available User IDs in the database", 
				value=str(users), 
				interactive=False,
				lines=6
			)
			
			generate_btn = gr.Button("Generate Recommendations", variant="primary")
			
			# User's purchased products will appear here
			purchased_products_html = gr.HTML(label="Previously Purchased Products")
		
		with gr.Column(scale=1):  # Right Side
			gr.Markdown("### Product Recommendations")
			recommendations_html = gr.HTML()
			gr.Markdown("### Performance Metrics")
			elapsed_time_html = gr.HTML()
	
	# Connect the button to the function
	generate_btn.click(
		get_user_recommendations,
		inputs=[user_id, weights],
		outputs=[purchased_products_html, recommendations_html, elapsed_time_html]
	)

if __name__ == "__main__":
	demo.launch()