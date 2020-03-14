import itertools
import numpy as np
from lightfm import LightFM


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
        super(WLightFM, self).__init__(**kwargs)

    def fit(self, interactions, **kwargs):
        """
        Fits the model just like LightFM do.
        Shadows item_factors and user_factors from item_embeddings and user_embeddings respectively
        Args:
            interactions: user-item sparse matrix with interactions
            **kwargs: Keyword arguments passed to lightfm.LightFM .fit() method
        Returns:
            Fitted model
        """
        super(WLightFM, self).fit(interactions, **kwargs)
        self.item_factors = self.item_embeddings
        self.user_factors = self.user_embeddings

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
        scores = super(WLightFM, self).predict(user_id, np.arange(matrix.shape[1]))
        if filter_already_liked_items:
            liked.update(matrix[user_id].indices)
        count = N + len(liked)
        if count < len(scores):
            ids = np.argpartition(scores, -count)[-count:]
            best = sorted(zip(ids, scores[ids]), key=lambda x: -x[1])
        else:
            best = sorted(enumerate(scores), key=lambda x: -x[1])
        return list(itertools.islice((rec for rec in best if rec[0] not in liked), N))
