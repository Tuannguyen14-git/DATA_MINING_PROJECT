import yaml
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.data.loader import load_data
from src.features.builder import build_tfidf
from src.mining.clustering import run_kmeans


def main():

    with open("configs/params.yaml") as f:
        config = yaml.safe_load(f)

    df = load_data(config["data_path"])

    texts = df[config["text_column"]].fillna("")

    X, vectorizer = build_tfidf(
        texts,
        config["tfidf"]["max_features"]
    )

    model, labels = run_kmeans(
        X,
        config["clustering"]["n_clusters"]
    )

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/tables", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

    joblib.dump(model, "outputs/models/kmeans_model.pkl")
    joblib.dump(vectorizer, "outputs/models/tfidf_vectorizer.pkl")

    df["cluster"] = labels
    df.to_csv("outputs/tables/clustering_results.csv", index=False)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X.toarray())

    plt.figure()
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels)
    plt.title("KMeans Clusters (PCA)")
    plt.savefig("outputs/figures/cluster_visualization.png")

    print("Pipeline completed")
    print("Results saved in outputs/")


if __name__ == "__main__":
    main()