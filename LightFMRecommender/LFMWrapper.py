import itertools
import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


class WLightFM(LightFM):
    def __init__(self, **kwargs):
        """
        WLightFM is a wrapper inherited from lightfm.LightFm
        This wrapper used to copy functionality of implicit.MatrixFactorizationBase class
        So, It reproduce LightFM implementation with MatrixFactorizationBase methods from benfred/implicit library
        Args:
            **kwargs: Keyword arguments passed to lightfm.LightFM .__init__() method
        Attributes:
            item_factors: Shadows item_factors name from MatrixFactorizationBase object
            user_factors: Shadows user_factors name from MatrixFactorizationBase object
        """
        self.item_factors = []
        self.user_factors = []
        self.interactions = None
        self.weights = None
        self.user_features = None
        self.item_features = None
        super(WLightFM, self).__init__(**kwargs)

    def fit_data(self, matrix, user_features=None, item_features=None):
        """
        Create datasets for .fit() method.
        Args:
            matrix: User-item interactions matrix (weighted)
            user_features: User-features pandas dataframe which index contains user_ids (crd_no)
            item_features:  Item-features pandas dataframe which index contains good_ids (plu_id)
        Returns:
            Model with fitted (mapped) datasets
        """
        matrix.sort_index(inplace=True)
        matrix.sort_index(inplace=True, axis=1)
        dataset = Dataset()
        dataset.fit((x for x in matrix.index),
                    (x for x in matrix.columns))
        interactions = pd.melt(matrix.replace(0, np.nan).reset_index(),
                               id_vars='index',
                               value_vars=list(matrix.columns[1:]),
                               var_name='plu_id',
                               value_name='rating').dropna().sort_values('index')
        interactions.columns = ['crd_no', 'plu_id', 'rating']
        self.interactions, self.weights = dataset.build_interactions([tuple(x) for x in interactions.values])

        if user_features is not None:
            user_features.sort_index(inplace=True)
            dataset.fit_partial(users=user_features.index,
                                user_features=user_features)
            self.user_features = dataset.build_user_features(
                        ((index, dict(row)) for index, row in user_features.iterrows()))
        else:
            self.user_features = None
        if item_features is not None:
            item_features.sort_index(inplace=True)
            dataset.fit_partial(items=item_features.index,
                                item_features=item_features)
            self.item_features = dataset.build_item_features(
                        ((index, dict(row)) for index, row in item_features.iterrows()))
        else:
            self.item_features = None

    def fit(self, **kwargs):
        """
        Fits the model just like LightFM do.
        Shadows item_factors and user_factors from item_embeddings and user_embeddings respectively
        Args:
            **kwargs: Keyword arguments passed to lightfm.LightFM .fit() method
        Returns:
            Fitted model
        """
        # warp-kos metric with weights not implemented in LightFM
        if self.loss == 'warp-kos':
            self.weights = None

        if self.interactions is None:
            raise AttributeError('Interactions not found, call .fit_data() method first')
        super(WLightFM, self).fit(interactions=self.interactions, sample_weight=self.weights,
                                  user_features=self.user_features, item_features=self.item_features,
                                  **kwargs)
        self.item_factors = self.get_item_representations()[1][:self.interactions.shape[1]]
        self.user_factors = self.get_user_representations()[1][:self.interactions.shape[0]]

    def recommend(self, user_id, matrix, filter_already_liked_items=False, N=10):
        """
        Recommend method that works just like MatrixFactorizationBase .recommend() method do.
        Args:
            user_id: User index (number) to make a recommendation for
            matrix: User-item sparse matrix
            filter_already_liked_items: Boolean if needed to filter items, that've been already bought
            N: Num recommendations
        Returns:
            list of tuples [(item_id_1, score_1), (item_id_2, score_2), ... , (item_id_N, score_N)]
            ordered by score descending
        """
        liked = set()
        num_threads = 1
        # with multiple threads performs better only when user_features or item_features specified
        if self.user_features is not None or self.item_features is not None:
            num_threads = 16
        scores = super(WLightFM, self).predict(user_id, np.arange(matrix.shape[1]), num_threads=num_threads,
                                               item_features=self.item_features, user_features=self.user_features)
        if filter_already_liked_items:
            liked.update(matrix[user_id].indices)
        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))
