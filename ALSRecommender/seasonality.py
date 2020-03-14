import pandas as pd
import numpy as np


def get_seasonality(bills, date_column='dates', group_column='level_4_name',
                    regular_only=False, promo_fact_column=None, kind='week'):
    """
    Calculates seasonality coefficient.
    Args:
        bills: Dataframe with bills
        date_column: Column that contains date
        group_column: Column, on which level we are calculating seasonality
        regular_only: Boolean flag to avoid promo influence
        promo_fact_column: column that contains boolean promo flag
        kind: supports two options: 'month' or 'week'
    Returns:
    Per week seasonality coefficient for each "group_column"
    """
    bills['year'] = pd.to_datetime(bills[date_column]).dt.year
    if kind == 'week':
        bills['period'] = pd.to_datetime(bills[date_column]).dt.week
        bounds = (1, 52)
    elif kind == 'month':
        bills['period'] = pd.to_datetime(bills[date_column]).dt.month
        bounds = (1, 12)
    # - Группируем по неделя-год, суммируем. Группируем по неделям, усредняем. (Если данные неравномерные)
    if not regular_only:
        num_per_week = bills.groupby([group_column, 'period', 'year'])[group_column].count().reset_index(name='num_sold')
        num_per_week = num_per_week.groupby([group_column, 'period'])['num_sold'].mean().reset_index(name='num_sold')
    else:
        # - Выбираем только регулярные продажи, считаем кол-во продаж и кол-во plu продававшихся регулярно на неделе
        num_per_week = bills[bills[promo_fact_column] == 0].groupby([group_column, 'period', 'year']).agg(
            {group_column: 'count', 'plu_name': 'nunique'})
        num_per_week = num_per_week.rename(columns={group_column: 'total_sold', 'plu_name': 'unique_plu'}).reset_index()
        # - Берем среднее по кол-ву рег. продаж и кол-ву рег. PLU по неделям между годами
        num_per_week = num_per_week.groupby([group_column, 'period'])[['total_sold', 'unique_plu']].mean().reset_index()
        # - Считаем кол-во регулярных продаж на кол-во рег. PLU (другими словами, если будет много товаров в категории
        # - На промо, то мы всё равно получим адекватную цифру.
        # - +10 - регуляризация
        num_per_week['num_sold'] = num_per_week['total_sold'] / (num_per_week['unique_plu']+10)
        num_per_week.drop(['total_sold', 'unique_plu'], axis=1, inplace=True)
    # - Делаем таблицу в которой есть все Категории и для каждого есть 52 недели
    new_table = pd.concat(
        [pd.DataFrame({group_column: x, 'period': [x + 1 for x in range(bounds[1])]})
         for x in bills[group_column].unique()])
    # - Добавляем туда фактические продажи и если продаж нет то заполняем нулями
    new_table = new_table.merge(num_per_week, on=[group_column, 'period'], how='left').fillna(0)
    # - Добавляем общее кол-во проданных PLU за всё время
    total_sold = new_table.groupby([group_column])['num_sold'].sum().reset_index(name='total_sold')
    new_table = new_table.merge(total_sold, on=group_column, how='left')
    # - Добавляем кол-во проданных на следующей и предыдущей неделе
    new_table['num_sold_prev'] = new_table.sort_values('period').groupby([group_column]).num_sold.shift(1)
    new_table['num_sold_next'] = new_table.sort_values('period').groupby([group_column]).num_sold.shift(-1)
    # - Обрабатываем граничные условия (52 и 1 неделя года)
    plu_52_week_sales = dict(new_table[new_table['period'] == bounds[1]].set_index([group_column])['num_sold'])
    plu_1_week_sales = dict(new_table[new_table['period'] == bounds[0]].set_index([group_column])['num_sold'])
    new_table.loc[new_table['period'] == bounds[0], 'num_sold_prev'] = \
        new_table[new_table['period'] == bounds[0]][group_column].map(
        lambda x: plu_52_week_sales[x])
    new_table.loc[new_table['period'] == bounds[1], 'num_sold_next'] = \
        new_table[new_table['period'] == bounds[1]][group_column].map(
        lambda x: plu_1_week_sales[x])
    # - Считаем скользящее среднее
    new_table['rolling_average'] = (new_table['num_sold_prev'] + new_table['num_sold'] + new_table['num_sold_next']) / \
                                   (new_table['total_sold'])
    # - Коэффициент сезонности - делим rolling_average на максимум - получаем распределние от 0 до 1
    new_table['seasonality'] = new_table['rolling_average'] / new_table.groupby(group_column)['rolling_average']\
                                                                       .transform(np.max)
    return new_table[[group_column, 'period', 'rolling_average', 'seasonality']]


