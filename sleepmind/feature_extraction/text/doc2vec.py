import logging
import os

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.utils import simple_preprocess

from sleepmind.base import BaseTransformer

logger = logging.getLogger(__name__)


class Doc2VecTransformer(BaseTransformer):
    """Doc2Vec.

    Notes: https://cs.stanford.edu/~quocle/paragraph_vector.pdf
    """

    def __init__(
        self,
        model="PV-DBOW",
        word_embeddings=False,
        window=8,
        alpha=.025,
        min_alpha=.001,
        epochs=10,
        vector_size=100,
        min_count=5,
        train=True,
        pretrained_model=None,
        n_jobs=1,
    ):
        """Instantiate Doc2VecTransformer object.

        Args:
            model (str): Either `PV-DBOW` or `PV-DM`. `PV-DM` considers word
                position.
            word_embeddings (bool): Whether to create word embeddings.
            vector_size (int): Vector size.
            min_count (int):
            alpha (float): Learning rate.
            min_alpha (float):
            epochs (int): Number of iterations.
            train (bool): Whether the model should be trained or re-using a
                previously trained model. If `False`, `serialized_model` should
                point to the location of a previously trained model.
            pretrained_model (str): Location of pre-trained model. If not `None`
                the argument `train` should be set to `False`.
            n_jobs (int):

        Attributes:
            clf (): Classifier.
            alpha_delta (float):

        Raises:
            ValueError: when mismatching between `train` and `pretrained_model`
                parameters.
        """
        if train and pretrained_model:
            raise ValueError(
                "The argument `train` should be set to `False` "
                "if a pre-trained model should be used."
            )
        self.model = model
        self.word_embeddings = 1 if word_embeddings else 0
        self.n_jobs = n_jobs
        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.epochs = epochs
        self.train = train
        self.pretrained_model = pretrained_model

        # attributes
        self.clf = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        if value in [
            0,
            1,
        ]:  # BUG: when instantiating the object in a pipeline and the passing it to cross_val_score
            self._model = value
            pass
        elif value not in ["PV-DBOW", "PV-DM"]:
            raise ValueError("`model` can be either `PV-DBOW` or `PV-DM`.")
        elif value == "PV-DBOW":
            self._model = 0
        else:
            self._model = 1

    def transform(self, X):
        """Infer vectors for new documents.

        Raises:
             FileNotFoundError: when pretrained model cannot be found in the
                specified location.
        """
        test_corpus = list(self._read_corpus(X, tokens_only=True))

        if not self.train:  # load pre-trained model
            if not os.path.exists(self.pretrained_model):
                FileNotFoundError(
                    "Could not find {}".format(self.pretrained_model)
                )
            self.clf = Doc2Vec.load(self.pretrained_model)
        results = [self.clf.infer_vector(doc) for doc in test_corpus]
        return np.array(results)

    def fit(self, X, y=None):
        """Train Doc2Vec model."""
        if self.train:
            # shuffling training corpus improve performance (in case corpus
            # wasn't already randomly ordered)
            np.random.shuffle(X)
            train_corpus = list(self._read_corpus(X))

            # train doc2vec
            self.clf = Doc2Vec(
                dm=self.model,
                dbow_words=self.word_embeddings,
                alpha=self.alpha,
                min_alpha=self.min_alpha,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                negative=5,
                epochs=self.epochs,
                workers=self.n_jobs,
            )
            self.clf.build_vocab(train_corpus)
            logger.info("Starting to train Doc2Vec")
            self.clf.train(
                train_corpus,
                total_examples=self.clf.corpus_count,
                start_alpha=self.alpha,
                end_alpha=self.min_alpha,
                epochs=self.clf.epochs,
            )
            logger.info("Completed Doc2Vec training")
            self.clf.delete_temporary_training_data(
                keep_doctags_vectors=True, keep_inference=True
            )
        return self

    @staticmethod
    def _read_corpus(corpus, tokens_only=False):
        """Cast corpus in a format that can be passed to Doc2Vec.

        Args:
            corpus (np.array): An array where each element is a document.
            tokens_only (bool): `True` is pre-processing the training dataset,
                `False` when dealing with new documents.

        Yields:
            (gensim.TaggedDocument) A document per time.
            """
        for document in corpus:
            for idx, line in enumerate(document):
                if tokens_only:  # for test dataset
                    yield simple_preprocess(line)
                else:  # for training data add tags
                    yield TaggedDocument(
                        words=simple_preprocess(line), tags=[idx]
                    )

    def save(self, filename):
        self.clf.save(filename)
