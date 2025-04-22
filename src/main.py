from pipeline import Pipeline

if __name__ == "__main__":
    
    model_name = "dmis-lab/biobert-base-cased-v1.1"

    pipeline = Pipeline(model_name)

    print(pipeline.get_function_embeddings())
