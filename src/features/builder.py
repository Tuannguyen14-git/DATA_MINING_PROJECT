from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(texts, max_features):
    texts = [str(t) if t is not None else "" for t in texts]   # thêm dòng này

    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)

    X = vectorizer.fit_transform(texts)
    return X, vectorizer