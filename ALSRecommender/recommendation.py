import numpy as np
import pandas as pd
import scipy.sparse as sparse
import datetime
import dateutil.relativedelta
from ALSRecommender.seasonality import get_seasonality_weekly
from pandas import DataFrame


class RecSys:
    def __init__(self, model, user_item_weighted_matrix, 
                 user_column_name='crd_no', good_id_column_name='plu_id'):
        self.model = model
        self.user_item_weighted_matrix = user_item_weighted_matrix
        self.user_column_name = user_column_name
        self.good_id_column_name = good_id_column_name

    def get_recommendations_dict(self, filter_already_liked, num_recommendations, recommend_all=True):
        index_crd_dict = self.user_item_weighted_matrix.reset_index()['index'].to_dict()
        user_item_sparse = sparse.csr_matrix(self.user_item_weighted_matrix)
        if recommend_all:
            recs = self.model.recommend_all(user_items=user_item_sparse, show_progress=False,
                                            filter_already_liked_items=filter_already_liked, N=num_recommendations)
        else:
            recs = []
            for i in range(self.user_item_weighted_matrix.shape[0]):
                recs_i = self.model.recommend(i, user_item_sparse,
                                              filter_already_liked_items=filter_already_liked, N=num_recommendations)
                recs.append([x[0] for x in recs_i])
        recs = pd.DataFrame(recs).reset_index().rename(columns={'index': self.user_column_name})
        recs[self.user_column_name] = recs[self.user_column_name].map(lambda x: index_crd_dict[x])
        # Для Item-Item Recommender, иногда получаются пустые рекоммендации для пользователей, нужно убрать.
        recs = recs[~recs[0].isnull()]
        recs['recs'] = recs.apply(lambda row_df: [int(row_df[x]) for x in range(num_recommendations)], axis=1)
        recs_dict = recs[[self.user_column_name, 'recs']].set_index(self.user_column_name).to_dict()['recs']
        return recs_dict

    def get_recommendations_proba(self, filter_already_liked, num_recommendations):
        id_plu_dict = self.user_item_weighted_matrix.T.reset_index()['index'].to_dict()
        index_crd_dict = self.user_item_weighted_matrix.reset_index()['index'].to_dict()
        user_item_sparse = sparse.csr_matrix(self.user_item_weighted_matrix)
        recs = []
        for i in range(self.user_item_weighted_matrix.shape[0]):
            rec = self.model.recommend(i, user_item_sparse,
                                       filter_already_liked_items=filter_already_liked, N=num_recommendations)
            recs.append(rec)
        recs = pd.concat([pd.DataFrame((x[0], x[1], i) for x in recs[i]) for i in range(len(recs))])
        recs.columns = ['plu_index', 'probability', 'user_index']
        # т.к ItemItemRecommender выплевывает свой скор (описан на Конфлюенсе, нормируем его на единицу)
        if self.model.__class__.__name__ == 'ItemItemRecommender':
            recs['probability'] = recs['probability'].apply(np.log)/recs['probability'].apply(np.log).max()
        # т.к LightFM выплевывает свой скор (описан на Конфлюенсе, нормируем его на единицу)
        if self.model.__class__.__name__ == 'WLightFM':
            recs['probability'] = (recs['probability']-recs['probability'].min()) \
                                         / (recs['probability'].max()-recs['probability'].min())
        recs[self.user_column_name] = recs['user_index'].map(lambda x: index_crd_dict[x])
        recs[self.good_id_column_name] = recs['plu_index'].map(lambda x: id_plu_dict[x])
        return recs[[self.user_column_name, self.good_id_column_name, 'probability']]

    def make_recommendations(self, bills, products, prices,
                             group_seasonality_column='level_2_name', bills_date_column='dates',
                             num_recommendations=1000, filter_already_liked=False, week_threshold=1):
        seas_df = get_seasonality_weekly(bills, date_column=bills_date_column, group_column=group_seasonality_column)
        seas_df = seas_df[seas_df['week'] == self.get_weeknum(threshold=week_threshold)]
        # Считаем рекоммендации с вероятностями
        recs = self.get_recommendations_proba(filter_already_liked=filter_already_liked,
                                              num_recommendations=num_recommendations)
        recs = recs.merge(products[[self.good_id_column_name, group_seasonality_column]],
                          how='left', on=self.good_id_column_name)

        prices = prices.loc[~prices['avg_price'].isnull()].copy()
        prices['avg_price'] = prices['avg_price'].astype(float)
        prices['avg_quantity'] = prices['avg_quantity'].astype(float)
        recs = recs.merge(seas_df,
                          on=group_seasonality_column, how='left').merge(prices,
                                                                         on=self.good_id_column_name, how='left')
        recs['expected_revenue'] = recs['probability'] * recs['seasonality'] * recs['avg_price'] * recs['avg_quantity']
        return recs[[self.user_column_name, self.good_id_column_name, 'probability',
                     'expected_revenue', 'avg_price', 'avg_quantity', 'seasonality']]

    @staticmethod
    def get_weeknum(threshold=1):
        return (datetime.date.today() + datetime.timedelta(days=7*threshold)).isocalendar()[1]


class Recommendations(DataFrame):
    _metadata = ['good_name_column_name', 'good_id_column_name', 'user_column_name']

    def __init__(self, df,
                 products=None,
                 user_column_name='crd_no', good_id_column_name='plu_id', good_name_column_name='plu_name',
                 **kwargs):
        """
        Recommendations class. Used for finalizing recommendations
        Args:
            df: Dataframe from RecSys.make_recommendations() method
            products: Dataframe that contains plu_id, plu_name, category_name
            user_column_name: Name of the column contains users
            good_id_column_name: Name of the column contains item ids
            category_column_name: Name of the column contains categories
            good_name_column_name: Name of the column contains item names
            **kwargs: Keyword arguments passed to pd.DataFrame
        """
        if products is not None:
            self.good_name_column_name = good_name_column_name
            self.good_id_column_name = good_id_column_name
            self.user_column_name = user_column_name
            df = df.merge(products, on=good_id_column_name)
        super(Recommendations, self).__init__(data=df, **kwargs)

    def __finalize__(self, other, method=None):
        """Propagate metadata from other to self """
        # merge operation: using metadata of the left object
        if method == 'merge':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.left, name, None))
        # concat operation: using metadata of the first object
        elif method == 'concat':
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other.objs[0], name, None))
        else:
            for name in self._metadata:
                object.__setattr__(self, name, getattr(other, name, None))
        return self

    @property
    def _constructor(self):
        return Recommendations

    def filter_private(self, private_label_names: list):
        """
        Method that filters non-private-label items from recommendations
        Args:
            private_label_names: List of private label names (searches inclusion in plu_name)
        Returns:
            Filtered Recommendations Dataframe
        """
        recs_pl = self[self[self.good_name_column_name].str.contains('|'.join(private_label_names))]
        return Recommendations(recs_pl.reset_index(drop=True)).__finalize__(self)

    def filter_price(self, min_price=0, price_column='avg_price'):
        """
        Method that filters items from recommendations by price
        Args:
            min_price: Minimum price in roubles
            price_column: Name of the column that contains price
        Returns:
            Filtered Recommendations Dataframe
        """
        recs_pl = self[self[price_column] > min_price]
        return Recommendations(recs_pl.reset_index(drop=True)).__finalize__(self)

    def filter_categories(self, bills, category_column='level_2_name'):
        """
        Method that filters category that never had been bought by user
        Args:
            bills: Bills dataframe. Have to contain category_column and self.user_column_name
            category_column: Column that contains category name
        Returns:
            Filtered Recommendations Dataframe
        """
        crd_bills = bills.groupby([category_column, self.user_column_name])[self.user_column_name]\
            .count().reset_index(name='count')
        return Recommendations(self.merge(crd_bills[[category_column, self.user_column_name]],
                                          on=[category_column, self.user_column_name])).__finalize__(self)

    def filter_inactive_cards(self, bills, month_threshold=2, date_column='dates'):
        """
        Method that filters old cards, to maximize response probability
        Args:
            bills: Bills dataframe. Have to contains date_column and self.user_column_name
            month_threshold: Card stuck in months
            date_column: Column that contains bill date
        Returns:
            Filtered Recommendations DataFrame
        """
        last_buy = bills.groupby([self.user_column_name])[date_column].max().reset_index(name='last_date')
        last_buy['last_date'] = pd.to_datetime(last_buy['last_date'])
        max_date = last_buy['last_date'].max()
        date_to_find = max_date - dateutil.relativedelta.relativedelta(months=month_threshold)
        last_buy = last_buy[last_buy['last_date'] >= date_to_find]
        return Recommendations(self.merge(last_buy[[self.user_column_name]], how='inner')).__finalize__(self)

    def recommend_for_groups(self, e_form_factor=0.8,
                             revenue_column='expected_revenue', category_column='level_4_name'):
        """
        Method that divides all recommendations into groups of communication channels
        Args:
            revenue_column: Expected revenue column name in Recommendations dataframe
            e_form_factor: Form factor: http://10.50.124.101:20001/pages/viewpage.action?pageId=36962551)
            category_column: Column that contains category
        Returns:
            Three dataframe by channel groups: 1: 1 item SMS, 2: 1 category SMS, 3: Multi-goods Email
        """
        max_revenue_recs = self[self.groupby([self.user_column_name])[revenue_column].transform(max) * e_form_factor
                                < self[revenue_column]]
        # Считаем кол-во товаров категории по категориям на каждую карточку
        cat_per_crd = max_revenue_recs.groupby([self.user_column_name,
                                                category_column])[revenue_column].agg(['count', 'sum']).reset_index()
        # Выделяем тех, кому нужно отправить 1 СМС с 1 товаром
        one_good = max_revenue_recs.merge(cat_per_crd[cat_per_crd
                                          .groupby([self.user_column_name])['count'].transform(sum) == 1],
                                          on=[self.user_column_name, category_column], how='inner')
        # Убираем их из общей выборки
        max_revenue_recs = max_revenue_recs.merge(one_good[self.user_column_name],
                                                  how='left', on=[self.user_column_name], indicator=True)
        max_revenue_recs = max_revenue_recs[max_revenue_recs['_merge'] == 'left_only'].drop('_merge', axis=1)

        # Выделяем тех, кому нужно отправить 1 СМС с 1 Категорией (Доля категории среди рекоммендаций > 0.7)
        # Считаем долю категории в общей ожидаемой выручке для карточки
        max_revenue_recs['category_share'] = \
            max_revenue_recs.groupby([self.user_column_name, category_column])[revenue_column].transform(sum)\
            / max_revenue_recs.groupby([self.user_column_name])[revenue_column].transform(sum)

        # Составляем список рекоммендаций с категориями
        one_cat = max_revenue_recs[max_revenue_recs['category_share'] >= 0.7]
        one_cat = one_cat.groupby([self.user_column_name, category_column])[revenue_column]\
            .sum().reset_index(name=revenue_column)

        # Убираем их из общей выборки
        max_revenue_recs = max_revenue_recs.merge(one_cat[self.user_column_name],
                                                  how='left', on=[self.user_column_name], indicator=True)
        max_revenue_recs = max_revenue_recs[max_revenue_recs['_merge'] == 'left_only'].drop('_merge', axis=1)
        # Ищем рекоммендации для оставшихся карточек которые не попали в предыдущие варианты (1 категория или 1 вещь)
        leftover_recs = self.merge(max_revenue_recs[self.user_column_name].drop_duplicates(),
                                   how='inner', on=self.user_column_name)
        # Убираем товары из одинаковых категорий (оставляем наибольшие по ожидаемой выручке)
        leftover_recs = leftover_recs.sort_values(revenue_column, ascending=False).drop_duplicates(
            [self.user_column_name, category_column])
        # Выбираем для каждой карточки топ-5 товаров с наибольшей ожидаемой выручкой
        multi_goods = leftover_recs.sort_values(revenue_column, ascending=False).groupby([self.user_column_name])[
                     [revenue_column, category_column,
                      self.user_column_name, self.good_name_column_name]].head(5)
        return one_good, one_cat, multi_goods
