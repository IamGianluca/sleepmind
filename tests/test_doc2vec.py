import numpy as np
import pytest

from sleepmind.feature_extraction.text.doc2vec import Doc2VecTransformer


CORPUS = np.array(
    [
        ["i am working on doc2vec to improve my predictions"],
        ["Hi, my name is Luca and I love python"],
        ["i am testing that this transformer works"],
        ["we @re not entirely sure about the outcome"],
        ["three cats were crossing the street"],
        ["i think it is going to start raining soon"],
    ]
)


def test_doc2vec_vector_size():
    """The Doc2VecEstimator should return a np.ndarray of shape
    (n_documents, vector_size).
    """
    vectors_size = 10
    clf = Doc2VecTransformer(
        vector_size=vectors_size,
        n_jobs=1,
        min_count=2,
        alpha=.025,
        min_alpha=.001,
        epochs=20,
    )
    clf.fit(X=CORPUS)
    result = clf.transform(X=CORPUS)
    assert result.shape == (CORPUS.shape[0], vectors_size)


def test_raise_error_when_train_is_not_false_and_serialized_model_is_passed():
    with pytest.raises(ValueError):
        Doc2VecTransformer(
            vector_size=10,
            n_jobs=10,
            train=True,
            pretrained_model="./some_location/model.pkl",
        )


# def test_load_pretrained_model():
# # given
# n_vector, vectors_size = CORPUS.shape[0], 10
# filename = './src/sleepmind/models/tests/model.d2v'
# clf = Doc2VecTransformer(vector_size=vectors_size, n_jobs=4, min_count=2,
# alpha=.025, min_alpha=.001, epochs=20, train=True)
# clf.fit(X=CORPUS)
# clf.save(filename=filename)

# # when
# pretrain_clf = Doc2VecTransformer(train=False, pretrained_model=filename)
# vectors = pretrain_clf.transform(X=CORPUS)

# # then
# assert vectors.shape == (n_vector, vectors_size)
