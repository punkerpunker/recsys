import numpy as np
import pandas as pd
import scipy.sparse as sparse
import random
import copy
import dateutil.relativedelta
from implicit.nearest_neighbours import bm25_weight, tfidf_weight, normalize


def filter_old_cards(df, month_threshold=1, date_column='dates', card_column='CRD_NO'):
    """
    Removes old cards. Take those cards date of last purchase of which is less than max_date - month_threshold
    Args:
        df: Dataframe with bills
        month_threshold: Max month threshold of last purchase for specific crd
        date_column: column that contains sold date
        card_column: Column that contains crd number

    Returns:
    Filtered DataFrame
    """
    df[date_column] = pd.to_datetime(df[date_column])
    max_date = df[date_column].max()
    date_to_find = max_date - dateutil.relativedelta.relativedelta(months=month_threshold)
    cards_last_date = df.groupby([card_column])[date_column].max().reset_index(name='date_max')
    df = df.merge(cards_last_date, on=card_column, how='left')
    return df[df['date_max'] > date_to_find].drop('date_max', axis=1)


def filter_rare_cards(df, rarity_num=5, card_column='CRD_NO', date_column='dates'):
    """
    Removes rare cards. Takes only cards which have more than rarity_num bills (different days)
    Args:
        df: Dataframe with bills
        rarity_num: Minimum num bills per crd
        card_column: Column that contains crd number
        date_column: Column that contains date
    Returns:
    Filtered DataFrame
    """
    # drop_duplicates - Чтобы оставить только уникальные пары карта-день
    # (смотреть сколько разных дней покупатель приходил)
    cards_num_bills = df.drop_duplicates([date_column, card_column])\
                        .groupby([card_column]).size().reset_index(name='bill_counts')
    df = df.merge(cards_num_bills, on=card_column, how='left')
    return df[df['bill_counts'] > rarity_num].drop('bill_counts', axis=1)


def filter_rare_goods(df, rarity_num=5, date_column='dates', plu_column='PLU_ID'):
    """
    Removes rare goods. Takes only specific goods, num per month sold of which more than rarity num.
    Args:
        df: Dataframe with bills
        rarity_num: Minimum average num per month sold
        date_column: column that contains sold date
        plu_column: column that contains plu number

    Returns:
    Filtered DataFrame
    """
    df['year_month'] = pd.to_datetime(df[date_column]).dt.to_period('M')
    plu_num_bills = df.groupby([plu_column, 'year_month']).size().reset_index(name='plu_counts')
    plu_num_bills_per_month = plu_num_bills.groupby(plu_column)['plu_counts'].mean().reset_index(name='mean_num')
    df = df.merge(plu_num_bills_per_month, on=plu_column, how='left')
    return df[df['mean_num'] > rarity_num].drop(['mean_num', 'year_month'], axis=1)


def filter_old_goods(df, month_threshold=1, date_column='dates', plu_column='PLU_ID'):
    """
    Removes old goods. Takes only that goods, last date of sold of which is less than a max_date - month_threshold
    Args:
        df: Dataframe with bills
        month_threshold:  Max month threshold of last sale of specific plu
        date_column: column that contains sold date
        plu_column: column that contains plu number

    Returns:
    Filtered DataFrame
    """
    df[date_column] = pd.to_datetime(df[date_column])
    max_date = df[date_column].max()
    date_to_find = max_date - dateutil.relativedelta.relativedelta(months=month_threshold)
    plu_last_date = df.groupby([plu_column])[date_column].max().reset_index(name='date_max')
    df = df.merge(plu_last_date, on=plu_column, how='left')
    return df[df['date_max'] > date_to_find].drop('date_max', axis=1)


def filter_by_quantile(df, plu_count_quantiles=(0.5, 0.99), cards_count_quantiles=(0.4, 0.99),
                       plu_column='PLU_ID', card_column='CRD_NO'):
    # Cut df by plu_count
    """
    Filter by plu and card quantiles
    Args:
        df: DataFrame with bills
        plu_count_quantiles: list-like. Upper and lower quantile threshold of num appearances of plu
        cards_count_quantiles: list-like. Upper and lower quantile threshold of num appearances of crd
        plu_column: Column that contains PLU_ID
        card_column: Column that contains CRD_NO
    Returns:
    Filtered dataframe
    """
    df = df[(df.groupby(plu_column)[plu_column].transform('size') >=
             df[plu_column].value_counts().quantile(plu_count_quantiles[0]))
            &
            (df.groupby(plu_column)[plu_column].transform('size') <=
             df[plu_column].value_counts().quantile(plu_count_quantiles[1]))]

    # Cut df by cards_count
    df = df[(df.groupby([card_column])[card_column].transform('size') >=
             df[card_column].value_counts().quantile(cards_count_quantiles[0]))
            &
            (df.groupby([card_column])[card_column].transform('size') <=
             df[card_column].value_counts().quantile(cards_count_quantiles[1]))]
    return df


class Matrix(pd.DataFrame):
    """
    Class that converts bills to matrix and applies weights, transformations and so on
    """
    def __init__(self, df,
                 plu_column='plu_id', card_column='crd_no', aggregation='count',
                 products=None, category_column='level_4_name', date_column='dates',
                 **kwargs):
        """
        Args:
            plu_column: string. Name of column which contains PLU
            card_column: string. Name of column which contains Card numbers
            values: string. Name of column which contains weights
            aggfunc: string. Name of aggregation function to use when constructing pivot
        """
        if df.shape[1] < 20:
            df = self.construct(bills=df,
                                plu_column=plu_column, card_column=card_column,
                                aggregation=aggregation,
                                products=products, category_column=category_column, date_column=date_column)
        super(Matrix, self).__init__(data=df, **kwargs)

    def construct(self, bills, plu_column, card_column, aggregation, products, category_column, date_column):
        bills = self._columns_to_lowercase(bills)

        if aggregation == 'count':
            matrix = bills.reset_index()\
                          .pivot_table(columns=plu_column, index=card_column,
                                       values='index', aggfunc='count') \
                          .fillna(0)

        elif aggregation == 'attenuation':
            if products is None:
                raise NameError('For attenuation weight you have to specify products dataframe')
            else:
                bills_with_weight = self.attenuation(bills, products,
                                                     plu_column, card_column,
                                                     category_column, date_column)
                matrix = bills_with_weight.pivot_table(columns=plu_column, index=card_column,
                                                       values='weight', aggfunc='sum')\
                                          .fillna(0)
        else:
            raise NotImplementedError('Aggregation %s not implemented!' % aggregation)
        crd_list = list(matrix.index.values)
        plu_list = list(matrix.columns)
        matrix.columns = plu_list
        matrix.index = crd_list
        return matrix

    @staticmethod
    def attenuation(bills, products,
                    plu_column, card_column, 
                    category_column, date_column):
        """
        Function applies attenuation weights to each purchase
        Args:
            bills: DataFrame with bills
            products: DataFrame with products
            plu_column: Column contains plu_id
            card_column: Column contains crd_no
            category_column: Column contains level_4_name (For Karusel)
            date_column: Column contains purchase date

        Returns:
        DataFrame contains column 'weight' with attenuation weights
        """
        bills_with_products = bills.merge(products[[category_column, plu_column]], on=plu_column, how='left')
        # Считаем раз в сколько дней в среднем покупают каждую категорию
        category_dates = bills_with_products.groupby([category_column, card_column])[date_column].agg(
            ['max', 'min', 'nunique']).reset_index()
        category_dates['days'] = (category_dates['max'] - category_dates['min']).dt.days
        # Считаем инвертированную частоту покупки товара пользователем(Кол-во товара на кол-во дней на продаже)
        category_dates['periodicity'] = (category_dates['days']) / (category_dates['nunique'] - 1)
        category_freq = category_dates[category_dates['periodicity'] < 365].groupby([category_column])[
            'periodicity'].mean().reset_index(name='periodicity')
        # Цепляем частоты к общему датафрейму
        bills_with_products_frequency = bills_with_products.merge(category_freq[[category_column, 'periodicity']],
                                                                  on=[category_column], how='left')
        # Ищем для каждого пользователя дату последней покупки
        users_last_dates = bills_with_products.groupby([card_column])[date_column].max().reset_index()
        users_last_dates.rename(columns={date_column: 'last_purchase_date'}, inplace=True)
        # Цепляем даты последних покупок к общему датафрейму
        full = bills_with_products_frequency.merge(users_last_dates, on=card_column, how='left')
        # Считаем "Старость" покупки
        full['oldness'] = (full['last_purchase_date'] - full[date_column]).dt.days
        full['weight'] = 1 / ((full['oldness'] * 10) / (full['periodicity']) + 1)
        return full[[plu_column, card_column, 'weight']]

    @property
    def _constructor(self):
        return Matrix

    def transform(self, method='no', clip_upper_value=100):
        """
        Function transforms every single value in matrix with specified rules
        Args:
            method: Transformation method (no, clip)
            clip_upper_value: clip upper value
        Returns:
            Transformed matrix
        """
        if method == 'no':
            return Matrix(self)
        elif method == 'clip':
            return Matrix(self.clip(upper=clip_upper_value))
        elif method == 'log':
            matr = pd.DataFrame(self)
            return Matrix(matr.apply(np.log).clip(0, clip_upper_value))

    @staticmethod
    def _columns_to_lowercase(df):
        df.columns = [x.lower() for x in df.columns]
        return df

    def apply_weights(self, weight='bm25', k=100, b=0.8):
        """
        Function apply weights to user-item matrix
        Args:
            weight: (bm25, tf-idf, normalize) - weight method
            k: K1 parameter for bm25
            b: B parameter for bm25

        Returns:
        Weighted user-item matrix
        """
        crd_list = list(self.index.values)
        plu_list = list(self.columns)

        if weight == 'bm25':
            matrix = pd.DataFrame(bm25_weight(sparse.csr_matrix(self.to_numpy(), dtype='float16'), B=b, K1=k).toarray())

        elif weight == 'tf-idf':
            matrix = pd.DataFrame(tfidf_weight(sparse.csr_matrix(self.to_numpy(), dtype='float16')).toarray())

        elif weight == 'normalize':
            matrix = pd.DataFrame(normalize(sparse.csr_matrix(self.to_numpy(), dtype='float16')).toarray())

        else:
            matrix = pd.DataFrame(sparse.csr_matrix(self.to_numpy()).toarray())
            
        matrix.columns = plu_list
        matrix.index = crd_list
        return Matrix(matrix)
