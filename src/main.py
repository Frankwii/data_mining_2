import numpy as np

from functools import reduce
from clustering import HDBSCAN, KMeans
from pipeline import Pipeline

def prettify_clustering(formatted_clustering: list[list[str]], pipeline: Pipeline) -> str:
    def cluster_separator(cluster_number: int):
        n = str(cluster_number + 1)
        return "\n" + "=" * 8 + f"CLUSTER {n}" + "=" * 8 + "\n"

    result = ""
    for n, cluster in enumerate(formatted_clustering):
        result += cluster_separator(n)
        result += "\n".join(cluster)
        result += f"\n\nAffected bacteria: {", ".join(pipeline.get_all_affected_bacteria(cluster))}"

    return result

def prettify_semantic_search(result: list[tuple[str, float]]) -> str:

    return "\n".join(map(lambda t: t[0], result))

if __name__ == "__main__":
    
    print("CLUSTERING FUNCTIONS AND SORTING THEM BY RARITY...")

    model_name = "dmis-lab/biobert-base-cased-v1.1"

    pipeline = Pipeline(model_name)

    function_embeddings = pipeline.get_function_embeddings()

    clusterer = HDBSCAN(min_cluster_size=2)
    # clusterer = KMeans(100)

    retriever = pipeline.instantiate_retriever(clusterer, "min-max")

    sorted_clusters = retriever.sort_clusters()

    formatted_clustering = clusterer.format_clustering(pipeline.get_functions(), sorted_clusters)

    print(prettify_clustering(formatted_clustering, pipeline))

    print("###################################################")
    print("###################################################")
    print("###################################################")

    print("SORTING BACTERIA BY RELEVANCE TO 'Flagellum'...")

    print(
        prettify_semantic_search(pipeline.bacterium_semantic_search("Flagellum"))
    )

