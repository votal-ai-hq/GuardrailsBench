"""tfidf-lr-v1: word + character n-gram TF-IDF into logistic regression.

Two independent models, one per stage, because the stages see different text and
fail differently: the input model never sees the response, and the output model
is the only thing standing between an ``output_only`` item and a leak.

Character n-grams are in the mix on purpose — they are what keeps leetspeak and
zero-width-joined text from falling off a cliff the way pure word features do.
They also make the model an eager learner of surface form, which is exactly the
behaviour the benign-mirror pool is there to charge for.
"""
from __future__ import annotations

from ..schema import LABEL_VIOLATING, Item
from .base import ScoreGuardrail

SEED = 1337


def _build_pipeline():
    # Imported lazily so that importing guardrailsbench.systems does not hard-depend
    # on scikit-learn for anyone who only wants the schema and metrics.
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                 min_df=2, sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                 min_df=2, sublinear_tf=True, lowercase=True)),
    ])
    clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced",
                             random_state=SEED)
    return Pipeline([("feat", features), ("clf", clf)])


class TfidfLRGuardrail(ScoreGuardrail):
    name = "tfidf-lr-v1"

    def __init__(self) -> None:
        super().__init__()
        self._in_model = None
        self._out_model = None

    def fit(self, items: list[Item]) -> TfidfLRGuardrail:
        y = [1 if it.label == LABEL_VIOLATING else 0 for it in items]
        if len(set(y)) < 2:
            raise ValueError("tfidf-lr-v1 needs both classes in the training split")
        self._in_model = _build_pipeline().fit([it.input_text for it in items], y)
        self._out_model = _build_pipeline().fit([it.response for it in items], y)
        return self

    def _p(self, model, text: str) -> float:
        if model is None:
            raise RuntimeError("tfidf-lr-v1 was scored before fit()")
        return float(model.predict_proba([text])[0][1])

    def input_score(self, item: Item) -> float:
        return self._p(self._in_model, item.input_text)

    def output_score(self, item: Item) -> float:
        return self._p(self._out_model, item.response)
