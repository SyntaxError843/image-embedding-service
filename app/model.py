from fastembed import ImageEmbedding

class EmbeddingModel:
    def __init__(self):
        self.model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")

    def embed(self, images):
        return list(self.model.embed(images))


# Singleton instance
embedding_model = EmbeddingModel()