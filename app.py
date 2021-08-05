import logging
import re
import sys
import time
from datetime import datetime, timezone

import awswrangler as wr
import numpy as np
import pandas as pd
from boto3 import client, resource
from dateutil import parser

pd.set_option("display.max_rows", None, "display.max_columns", None)
pd.set_option('expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)


def logger_initiate():

    console_formatter = logging.Formatter(
        '%(levelname)s - %(asctime)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.propagate = 0
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
    stdout_handler.setFormatter(console_formatter)

    return logger


def delete_prev_files(df_result, output_bucket_name, output_prefix):
    s3_res = resource('s3')
    unique_sec_id = df_result['Sec'].unique()
    bucket_res = s3_res.Bucket(output_bucket_name)

    for sec in unique_sec_id:
        available_key = []
        # Find available date which are already existing in output location
        prefix = f'{output_prefix}{sec}/'
        object_summary = bucket_res.objects.filter(Prefix=prefix)
        for objects in object_summary:
            available_key.append(objects.key)

        available_dates = re.findall(r'(\d{4}-\d{2}-\d{2})', ' '.join(
            available_key))  # extract date folders from s3 path

        #  Compare output vs existing s3 date folder dataframe and find out
        #  date folder which needs to be deleted
        df_temp_dt_existing = pd.DataFrame(available_dates, columns=[
            'CalenderDate'])
        df_temp_dt_output = df_result[['CalenderDate']].copy()

        df_temp_dt_output['CalenderDate'] = df_temp_dt_output[
            'CalenderDate'].astype(str)
        df_dt_diff = df_temp_dt_existing.merge(df_temp_dt_output,
                                               how='outer', indicator=True).loc[
            lambda x: x['_merge'] == 'left_only']
        df_dt_diff = df_dt_diff[['CalenderDate']].drop_duplicates()

        if df_dt_diff.shape[0] > 0:
            dt_diff_list = df_dt_diff['CalenderDate'].to_list()
            for dt in dt_diff_list:
                bucket_res.objects.filter(Prefix=f"{prefix}{dt}/").delete()

            console_logger.info(f'Previous Folders will be deleted for these '
                                f'dates:{dt_diff_list}')
        else:
            console_logger.info('Nothing to delete')
            pass


def s3_obj_listing(bucket_name, prefix, lambda_start_time, lambda_event_time):
    s3_conn = client('s3')
    s3_result = s3_conn.list_objects_v2(Bucket=bucket_name, Prefix=prefix,
                                        Delimiter="/")
    if 'Contents' not in s3_result:
        console_logger.info('No object inside input path')
        sys.exit(1)

    file_datetime_list = []
    obj_key = []
    try:
        for key in s3_result['Contents']:
            last_modified_time = key['LastModified']
            file_key = key['Key']
            file_datetime_dict = {'key': file_key, 'last_modified_time':
                last_modified_time}
            obj_key.append(file_key)
            file_datetime_list.append(file_datetime_dict)

        while s3_result['IsTruncated']:
            continuation_key = s3_result['NextContinuationToken']
            s3_result = s3_conn.list_objects_v2(Bucket=bucket_name,
                                                Prefix=prefix, Delimiter="/",
                                            ContinuationToken=continuation_key)
            for key in s3_result['Contents']:
                last_modified_time = key['LastModified']
                file_key = key['Key']
                file_datetime_dict = {'key': file_key, 'last_modified_time':
                    last_modified_time}
                obj_key.append(file_key)
                file_datetime_list.append(file_datetime_dict)

    except Exception as error:
        console_logger.error(error)

    if bucket_name == 'sg-output':
        df_ = pd.DataFrame(file_datetime_list)
        df_ = df_.sort_values(by='last_modified_time', ascending=False)

        #  Consider trade files which came between lambda invoke timestamp and
        #  lambda execution start time
        df_ = df_.query("last_modified_time >= @lambda_event_time & "
                        "last_modified_time <= @lambda_start_time")
        file_list = df_['key'].to_list()

        #  append s3://bucket name to the object key
        s3_substring = f's3://{bucket_name}/'
        file_list = list(map(lambda s: s3_substring + s, file_list))
        if len(file_list) == 0:
            console_logger.info(f'No file was found in s3 between time '
                          f'range:{lambda_event_time} - {lambda_start_time}. '
                          'Terminating the application.')
            sys.exit(1)

        return file_list
    else:
        return obj_key


def get_input(input_bucket_name, input_prefix, lambda_start_time,
              lambda_event_time):
    column_names = ['Sec', 'trade_id', 'audit_version', 'TradeDate',
                    'SettleDate', 'Notional', 'BondMktType', 'TradeStatus',
                    'Audit_DateTime', 'Ccy']
    file_list = s3_obj_listing(input_bucket_name, input_prefix,
                               lambda_start_time, lambda_event_time)
    console_logger.info(f'Trade files which were considered: {file_list}')
    """
    This is to filter out the files specific to secid which was passed by 
    lambda event.For this partitioning should be implemented, else the 
    operation will be expensive or secid needs to be present in object prefix.

    secid_files_list = [file for file in file_list if any(secid in file for
                                                          secid in sec_ids)]
    df = wr.s3.read_parquet(secid_files_list, columns = column_names, 
                                                          chunked=True)
    """
    df = wr.s3.read_parquet(file_list, columns=column_names)
    #  df = df[df['Sec'].isin(sec_ids)]
    console_logger.info('Input file dataframe creation completed successfully')
    return df


def calculate_settlement_balance(df_latest_version):
    try:
        df_latest_version.drop(['BondMktType', 'audit_version'], inplace=True,
                               axis=1)

        start_dt = df_latest_version[['TradeDate', 'SettleDate']].min().min()
        end_dt = df_latest_version[['TradeDate', 'SettleDate']].max().max()

        df_temp = pd.DataFrame(pd.date_range(start=start_dt,
                                             end=end_dt)).rename(
            columns={0: 'SettleDate'})
        df_temp = df_temp.merge(df_latest_version['Sec'].drop_duplicates(),
                                how='cross')

        df_settle_bal = df_latest_version.groupby(['Sec', 'SettleDate']).sum(
        ).cumsum().reset_index().merge(df_temp, on=['Sec', 'SettleDate'],
                                       how='outer')

        df_settle_bal = df_settle_bal.sort_values(['Sec', 'SettleDate'])

        df_settle_bal['Notional'] = df_settle_bal[['Sec', 'Notional']].groupby(
            'Sec').fillna(method='ffill')

        df_settle_bal.loc[df_settle_bal['Notional'].isna(), 'Notional'] = 0
        df_settle_bal = df_settle_bal.rename(
            columns={'Notional': 'BalSettledate'})
        df_settle_bal.drop('trade_id', inplace=True, axis=1)
        df_settle_bal.rename(columns={"SettleDate": "CalenderDate"},
                             inplace=True)
        console_logger.info(
            'Balance settlement calculation completed successfully')
        return df_settle_bal, start_dt, end_dt

    except Exception as error:
        console_logger.error(error)


def calculate_trade_balance(df):
    try:
        df_latest_version = df.groupby(['Sec', 'trade_id']) \
            .max().reset_index()  # Consider latest version only

        #  Consider BondMktType to decide +ve or -ve notional value
        df_latest_version.loc[(df_latest_version['BondMktType'] != 'INIT') & (
                df_latest_version['BondMktType'] != 'REOPEN'), 'Notional'] *= -1

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        #  Condition to check tradestatus =ver and audit_datetime <= current
        #  datetime
        df_latest_version = df_latest_version.query("TradeStatus == 'VER' & "
                                                    "Audit_DateTime <= @now")
        # currency !=usd

        #  Convert tradedate and settledate to date type
        df_latest_version['TradeDate'] = pd.to_datetime(df_latest_version[
                                                            'TradeDate'],
                                                        format='%Y%m%d')
        df_latest_version['SettleDate'] = pd.to_datetime(df_latest_version[
                                                             'SettleDate'],
                                                         format='%Y%m%d')
        df_currency = df_latest_version[['Sec', 'Ccy']].copy()
        df_currency.drop_duplicates(inplace=True)
        sec_currency = dict(zip(df_currency.Sec, df_currency.Ccy))
        #  Aggregation for getting the sum and populate missing dates till
        #  max date
        df_agg = df_latest_version.groupby(['Sec', 'TradeDate'])[
            'Notional'].sum().groupby('Sec').cumsum().reset_index(
            level=0).resample('D').pad().reset_index()
        console_logger.info(
            'Balance trade calculation completed successfully')
        return df_agg, df_latest_version, sec_currency

    except Exception as error:
        console_logger.error(error)


def flatten_data(row):
    #  Extract even filed and secid from each dataframe row
    #  and insert secid into event field
    events_filed = row[0]['Events']['SEVENT'].tolist()
    sec_id = row[1]
    events_filed = [dict(item, **{'secid': sec_id}) for item in events_filed]
    return events_filed


def apply_xnl_condition(row):
    today = datetime.today().strftime('%Y%m%d')
    today = datetime.strptime(today, '%Y%m%d')
    xnl_date = row['Date']
    #  Convert int date to date type
    xnl_date = datetime(year=int(str(xnl_date)[0:4]), month=int(str(xnl_date)[
                                        4:6]), day=int(str(xnl_date)[6:8]))
    #  Condition check
    if (row['Type'] == 'XNL') & (xnl_date <= today):
        return 'Y'
    else:
        return 'N'


def xnl_calculation(bond_trade_files):
    try:
        df_xnl = wr.s3.read_parquet(bond_trade_files)

        df_xnl = df_xnl[['Issue', 'master_sec_id']]
        df_xnl['Issue'] = df_xnl['Issue'].apply(pd.Series)
        df_xnl['Issue'] = df_xnl.apply(flatten_data, axis=1)
        df_xnl.drop(['master_sec_id'], inplace=True, axis=1)
        df_xnl = df_xnl.explode('Issue')
        print(df_xnl)
        df_xnl['comments'] = df_xnl['Issue'].apply(apply_xnl_condition)
        df_xnl = df_xnl.query("comments == 'Y'")
        df_xnl = df_xnl['Issue'].apply(pd.Series)
        df_xnl = df_xnl[['secid', 'Date', 'Amount']]
        df_xnl = df_xnl.groupby(['secid', 'Date'])['Amount'].agg(
            'sum').reset_index()
        console_logger.info(
            'XNL calculation completed successfully')

    except Exception as error:
        console_logger.error(error)


def upload_to_s3(df_result, output_bucket_name, output_prefix):
    delete_prev_files(df_result, output_bucket_name, output_prefix)
    try:
        df_split_list = np.array_split(df_result, df_result.shape[0])
        i = 0
        for each_df in df_split_list:
            calender_dt = str(each_df.at[i, 'CalenderDate'])[0:10]
            sec_id = each_df.at[i, 'Sec']
            i = i + 1
            each_df.to_parquet(f's3://{output_bucket_name}/{output_prefix}'
                               f'{sec_id}/{calender_dt}/{sec_id}.parquet',
                               index=False)
        console_logger.info(
            'Uploading output files into s3 completed successfully')
    except Exception as error:
        console_logger.error(error)


def currency_check(row):
    print(sec_currency_dict)
    print(row)
    if sec_currency_dict[row[1]] == 'USD':
        trade_usd_val = row[2]
        settle_usd_val = row[3]
        trade_non_usd_val = 0
        settle_non_usd_val = 0
    else:
        trade_usd_val = 0
        settle_usd_val = 0
        trade_non_usd_val = row[2]
        settle_non_usd_val = row[3]

    return pd.Series([trade_usd_val, settle_usd_val, trade_non_usd_val,
                      settle_non_usd_val])


def lambda_handler(event, context):
    global console_logger, sec_currency_dict

    # initiate logger
    console_logger = logger_initiate()
    console_logger.setLevel(logging.INFO)

    # declare variables
    bond_security_bucket = event['Records'][0]['s3']['bucket']['name']
    file_name = event['Records'][0]['s3']['object']['key']
    sec_id = file_name[0:4]
    lambda_event_time = event['Records'][0]['eventTime']
    lambda_event_time = parser.parse(lambda_event_time)
    lambda_event_time = lambda_event_time.replace(tzinfo=timezone.utc)
    put_event_count = len(event['Records'])
    bond_trade_files = []
    bond_trade_file_path = 'parquet_output/bond_security'

    for each_file in event['Records']:
        object_key = each_file['s3']['object']['key']
        bond_trade_files.append(f's3://{bond_security_bucket}/'
                                f'{bond_trade_file_path}/' + object_key)
    console_logger.info('Bond security files to be considered for xnl '
                        f'calculation:{bond_trade_files}')

    time.sleep(1)
    lambda_start_time = datetime.now(timezone.utc)
    console_logger.info(f'Trade files consideration timeframe'
                        f':{lambda_event_time} to {lambda_start_time}')
    input_bucket_name = 'sg-output'
    input_prefix = 'parquet_output/bond_trade/'
    output_bucket_name = 'athena-output-glue'
    output_prefix = 'output/'

    # call functions
    df = get_input(input_bucket_name, input_prefix, lambda_start_time,
                   lambda_event_time)
    df_agg, df_latest_version, sec_currency_dict = calculate_trade_balance(df)
    sec_currency_dict = {3136: 'INR'}
    df_settle_bal, start_dt, end_dt = calculate_settlement_balance(
        df_latest_version)

    df_agg = df_agg.groupby("Sec").apply(
        lambda x: x.set_index("TradeDate").reindex(
            pd.date_range(start_dt, end_dt, freq="D")).reset_index().ffill()
    ).reset_index(drop=True)

    df_agg.rename({"index": "CalenderDate", "Notional": "BalTradedate"},
                  inplace=True,
                  axis=1)
    df_agg['Sec'] = df_agg['Sec'].map(np.int64)
    df_result = pd.merge(df_agg, df_settle_bal, on=['Sec', 'CalenderDate'])

    df_result[['BalTradedate_nonusd', 'BalSettledate_nonusd']] = 0
    df_result[['BalTradedate', 'BalSettledate', 'BalTradedate_nonusd',
               'BalSettledate_nonusd']] = df_result.apply(currency_check,
                                                          axis=1)
    print(df_result)
    upload_to_s3(df_result, output_bucket_name, output_prefix)
    xnl_calculation(bond_trade_files)
