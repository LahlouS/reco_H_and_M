from ALS_training_lib import load_interaction_matrix
from implicit.als import AlternatingLeastSquares
from datetime import datetime

def generate_time_postfix():
	return datetime.now().strftime("%Y%m%d_%H%M%S")

outfile = f"model_weights_{generate_time_postfix()}"
filepath_in = "/home/slahlou/Documents/recoGnomon/dataset/collaborative-filtering-data/interaction_matrix.npz"
filepath_out = "/home/slahlou/Documents/recoGnomon/program-v1/weights/" + outfile


interaction_matrix = load_interaction_matrix(filepath_in)

model = AlternatingLeastSquares(factors=64, regularization=0.05, alpha=2.0)
model.fit(interaction_matrix)



model.save(filepath_out)
print('LOG: weights saved succesfully at', filepath_out)