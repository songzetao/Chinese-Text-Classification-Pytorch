import numpy as np
file_path="./THUCNews/data/embedding_SougouNews.npz"
poem=(np.load(file_path))['embeddings']

print(poem)