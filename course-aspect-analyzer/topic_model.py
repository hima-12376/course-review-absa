from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class TopicModel:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_embeddings(self, texts):
        return self.model.encode(texts, show_progress_bar=True)

    def cluster_reviews(self, embeddings, n_clusters=5):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        return labels