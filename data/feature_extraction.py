from sklearn.feature_extraction.text import TfidfVectorizer

from .pre_processing import preprocessor, tokenizer


def tf_idf_vectorizer(x_tr, ngram_range=(1, 1), vocabulary=None, binary=False, use_idf=True, smooth_idf=True):
    tf_idf = TfidfVectorizer(preprocessor=preprocessor,
                             tokenizer=tokenizer,
                             ngram_range=ngram_range,
                             vocabulary=vocabulary,
                             binary=binary,
                             use_idf=use_idf,
                             smooth_idf=smooth_idf)
    x_tr = tf_idf.fit_transform(x_tr)

    return x_tr, tf_idf
