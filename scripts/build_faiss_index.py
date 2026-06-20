# import faiss
# import numpy as np

# embeddings = np.load("artifacts/review_embeddings.npy").astype("float32")
# dim = embeddings.shape[1]

# # Vectors are already L2-normalized, so inner product == cosine similarity
# index = faiss.IndexFlatIP(dim)
# index.add(embeddings)
# faiss.write_index(index, "artifacts/faiss_index.bin")
# print(f"Indexed {index.ntotal} vectors of dim {dim}")


import faiss
import numpy as np

# 1. Build the Text (Review) Index
txt_embeddings = np.load("artifacts/review_embeddings.npy").astype("float32")
txt_dim = txt_embeddings.shape[1]
txt_index = faiss.IndexFlatIP(txt_dim)
txt_index.add(txt_embeddings)
faiss.write_index(txt_index, "artifacts/faiss_text_index.bin")
print(f"✅ Indexed {txt_index.ntotal} TEXT vectors")

# 2. Build the Image Index
img_embeddings = np.load("artifacts/image_embeddings.npy").astype("float32")
img_dim = img_embeddings.shape[1]
img_index = faiss.IndexFlatIP(img_dim)
img_index.add(img_embeddings)
faiss.write_index(img_index, "artifacts/faiss_image_index.bin")
print(f"✅ Indexed {img_index.ntotal} IMAGE vectors")