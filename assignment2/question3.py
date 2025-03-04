import numpy as np

# Calculate cosine similarity.
# Cosine similarity is defined as the dot product of the two vectors,
# divided by the product of the magnitudes of the vectors.
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# Calculate Euclidean distance.
# Euclidean distance is the straight-line distance between two points
# in n-dimensional space.
def euclidean_distance(A, B):
    return np.linalg.norm(A - B)

# Declare two fixed, three-dimensional vectors.
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Compute cosine similarity
cs = cosine_similarity(v1, v2)
# Compute Euclidean distance
ed = euclidean_distance(v1, v2)

# Print results
print("Cosine Similarity:", cs)
print("Euclidean Distance:", ed)
