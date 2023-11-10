import awswrangler as wr
import pandas as pd
import numpy as np
import boto3

boto3.setup_default_session(region_name="us-east-1")

tables = ['p2p',
          'amm',
          'coin_deposits',
          'coin_withdrawals',
          'fiat_deposits',
          'fiat_withdrawals']

freqs = ['weekly', 'monthly']

def load_data(sql, db):
    return wr.athena.read_sql_query(sql=sql, database=db)


def numeric_object_2_float32(df):
    object_columns = df.select_dtypes(['object']).columns.tolist()
    type_dict = {obj_col: 'float32' for obj_col in object_columns}
    return df.astype(type_dict)


def feature_combination(dfs, freq='weekly', how='outer'):
    left_df = dfs[0]
    for right_df in dfs[1:]:
        left_df = pd.merge(left_df,
                           right_df,
                           how,
                           ['user_id', f'{freq}_report'])
    return left_df


def feature_processing(args=None):
    weekly_dfs = [load_data(f'select * from weekly_{table}', 'feature_stores') for table in tables]
    for wdf in weekly_dfs:
        wdf.drop(['week'], axis=1, inplace=True)
        wdf = numeric_object_2_float32(wdf)

    weekly_combination_df = feature_combination(weekly_dfs)
    weekly_combination_df['monthly_report'] = weekly_combination_df['weekly_report'] - pd.offsets.MonthBegin(1)

    monthly_dfs = [load_data(f'select * from monthly_{table}', 'feature_stores') for table in tables]
    for mdf in monthly_dfs:
        mdf.drop(['month'], axis=1, inplace=True)
        mdf = numeric_object_2_float32(mdf)

    monthly_combination_df = feature_combination(monthly_dfs, 'monthly')
    week_month_combination_df = feature_combination([weekly_combination_df,
                                                     monthly_combination_df],
                                                    'monthly',
                                                    'left')

    query = """
            SELECT user_id,
                   doc_country,
                   is_cloned,
                   created_at,
                   status,
                   current_country_code,
                   totu_achieved_at_glue,
                   username,
                   gender
            FROM users
            """
    user_demographic_df = load_data(query, 'feature_stores')

    user_demographic_df['label'] = user_demographic_df['is_cloned'].astype(int)

    campaign_users_df = user_demographic_df[(user_demographic_df.created_at >= '2022-06-01')
                                            & (user_demographic_df.created_at <= '2022-10-31')
                                            & (user_demographic_df.status == 'active')]

    campaign_feature_df = week_month_combination_df[(week_month_combination_dff.monthly_report >= '2022-06-01') 
                                                    & (week_month_combination_dff.monthly_report <= '2022-10-31')]

    campaign_feature_df['weekly_rank_record'] = (campaign_feature_df
                                                .groupby(['user_id', 'monthly_report'])['weekly_report']
                                                .rank(method='first', ascending=False))

    campaign_feature_df['monthly_rank_record'] = (campaign_feature_df
                                                 .groupby('user_id')['monthly_report']
                                                 .rank(method='first', ascending=False))

    campaign_feature_df['weekly_rank_record'] = pd.to_numeric(campaign_feature_df['weekly_rank_record'],
                                                              downcast='integer')

    campaign_feature_df['monthly_rank_record'] = pd.to_numeric(campaign_feature_df['monthly_rank_record'],
                                                               downcast='integer')


    # final_df = campaign_feature_df[(campaign_feature_df.weekly_rank_record == 1)
    #                                & (campaign_feature_df.monthly_rank_record == 1)]

    final_feature = pd.merge(campaign_feature_df,
                             campaign_users_df,
                             how='inner',
                             on=['user_id'])

    cats_columns = final_feature.select_dtypes(exclude=np.number).columns.tolist()

    for col in cats_columns:
        final_feature[col].fillna('TBD', inplace=True)
        final_feature[col] = final_feature[col].astype('category')

    final_feature.fillna(0, inplace=True)
    
    bucket = 's3://mlops-feature-stores/feature-ready/cloned-user-detection.parquet'
    wr.s3.to_parquet(df=data_prepared, path=bucket, dtype={cat: 'string' for cat in cats})

    return

if __name__ == "__main__":
    try:
        feature_processing()
    except Exception as e:
        raise Exception(e)