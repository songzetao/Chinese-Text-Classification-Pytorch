import numpy as np
import oneflow as torch
dataset = 'THUCNews'
embedding = 'embedding_SougouNews.npz'
tensor = torch.tensor(np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32')) if embedding != 'random' else None                                       # 预训练词向量
print(tensor)