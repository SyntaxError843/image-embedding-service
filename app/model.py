from fastembed import ImageEmbedding, TextEmbedding

class EmbeddingModel:
    def __init__(self):
        # Image model
        self.image_model = ImageEmbedding(
            model_name="Qdrant/clip-ViT-B-32-vision"
        )

        # Text model
        self.text_model = TextEmbedding()  # defaults to BAAI/bge-small-en-v1.5

    def embed_images(self, image_paths):
        return list(self.image_model.embed(image_paths))

    def embed_texts(self, texts):
        return list(self.text_model.embed(texts))


# Singleton instance
embedding_model = EmbeddingModel()