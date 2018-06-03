from sklearn.preprocessing import LabelEncoder


class ModifiedLabelEncoder(LabelEncoder):
    # TODO: deprecate as soon as sklearn 0.20 is released

    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        return super().transform(y).reshape(-1, 1)
