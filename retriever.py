import numpy as np
import torch
import pickle
import os
import scipy.sparse as sp
import math
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis
from scipy.spatial.distance import cdist

doc_input_raw = np.load('ori_input.npy', allow_pickle=True)
doc_embedding_raw = np.load('save_doc.npy', allow_pickle=True)

# Step 1: Collect all arrays from doc_input_raw
all_chunks = [arr for arr in doc_input_raw]  # Each of shape (B_i, 288, 139, 3)

# Step 2: Stack along batch dimension
stacked = np.concatenate(all_chunks, axis=0)  # Shape: (3893, 288, 139, 3)

# Step 3: Transpose to put 139 before 288: (B, 139, 288, 3)
stacked = stacked.transpose(0, 2, 1, 3)  # Now shape: (3893, 139, 288, 3)

# Step 4: Merge first and second dims: (3893×139, 288, 3)
doc_input = stacked.reshape(-1, 288, 3)


all_chunks = [arr for arr in doc_embedding_raw]  # Each arr is of shape (16, 288, 139, 3) or (5, 288, 139, 3)

# Concatenate along axis 0 (batch)
stacked = np.concatenate(all_chunks, axis=0)  # Shape: (243*16 + 5, 288, 139, 3)

# Step 2: Reshape to ((243*16 + 5)*139, 288, 3)
# B, N, T, C = stacked.shape
doc_embedding = stacked.reshape(-1, 36, 96)



query_input_raw = np.load('query_input.npy', allow_pickle=True)
query_embedding_raw = np.load('query_doc.npy', allow_pickle=True)
results =[]

def retrieve_id(doc_embedding, query_embedding, topk=3, pca_dim=128):
    doc_embedding = np.asarray(doc_embedding, dtype=np.float32).reshape(doc_embedding.shape[0], -1)
    query_embedding = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)

    pca = PCA(n_components=pca_dim)
    doc_pca = pca.fit_transform(doc_embedding)          # [N, pca_dim]
    query_pca = pca.transform(query_embedding)          # [1, pca_dim]


    cov = np.cov(doc_pca, rowvar=False) + np.eye(pca_dim) * 1e-5
    inv_cov = np.linalg.inv(cov)

    distances_Maha = np.array([
        mahalanobis(doc_pca[i], query_pca[0], inv_cov)
        for i in range(doc_pca.shape[0])
    ])
    distances_l2 = cdist(doc_embedding, query_embedding, metric='euclidean').squeeze()  # shape: [N]
    distances = distances_l2+distances_Maha/3
    topk_indices = np.argsort(distances)[:topk]
    topk_distances = distances[topk_indices]
    return topk_indices.tolist(), topk_distances.tolist()

for q in range(11): #11 for Nottingham dataset and 7 for Glasgow dataset
    query_input = query_input_raw[-1][-1,:,q,:]
    query_input = np.expand_dims(query_input, axis=0)
    query_embedding = query_embedding_raw[-1][0,q,:,:]
    query_embedding = np.expand_dims(query_embedding, axis=0)

    topk_idx, topk_dists = retrieve_id(doc_embedding, query_embedding, topk=1)
    for i, (idx, dist) in enumerate(zip(topk_idx, topk_dists), 1):
        print(f"Top-{i} doc id: {idx}, Loss distance: {dist:.4f}")
        tid, sid = divmod(idx, 139)  # tid is temporal id, sid is the spatial id, 139 is the number of the source nodes
        results.append((tid, sid))

results = np.array(results)

# save for OpenAI API usage
np.savetxt("retrieve_results.csv", results, fmt="%d", delimiter=",", header="tid,sid", comments="")
print("Saving completed: results.csv")