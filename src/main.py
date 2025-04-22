import numpy as np

from clustering import KMeans
from pipeline import Pipeline

if __name__ == "__main__":
    
    model_name = "dmis-lab/biobert-base-cased-v1.1"

    pipeline = Pipeline(model_name)

    function_embeddings = pipeline.get_function_embeddings()

    clusterer = KMeans(100)

    for cluster in clusterer.cluster_functions_formatted(function_embeddings):
        print("\n".join(cluster))
        print("================")
