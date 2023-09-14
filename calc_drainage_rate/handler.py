from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from cognite.client import CogniteClient
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess


# from statsmodels.tsa.seasonal import seasonal_decompose


def handle(client: CogniteClient, data: dict) -> pd.DataFrame:
    """Calculate drainage rate per timestamp and per day from tank,
    using Lowess filtering on volume percentage data from the tank.
    Large positive derivatives of signal are excluded to ignore
    human interventions (filling) of tank.
    Data of drainage rate helps detecting leakages.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        pd.DataFrame: dataframe with drainage rate and trend (derivative)
    """
    # STEP 0: Unfold data
    tank_volume = data["tank_volume"]
    derivative_value_excl = data["derivative_value_excl"]
    start_date = pd.to_datetime(data["start_date"], format="%Y-%m-%d-%H-%M-%S")
    end_date = start_date + timedelta(days=data["tot_days"])
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]
    data_set_id = data["data_set_id"]

    # STEP 1: Load time series from name and aggregate

    ts_in = client.time_series.search(name=ts_input_name)  # find time series by name
    ts_in_extid = ts_in[0].external_id  # extract its external id
    df_cdf = client.time_series.data.retrieve(
        external_id=ts_in_extid, aggregates="average", granularity="1m", start=start_date, end=end_date
    )  # load time series by external id

    df = df_cdf.to_pandas()
    df = df.rename(columns={ts_in_extid + "|average": ts_input_name})

    # STEP 2: Filter signal
    df["time_sec"] = (
        df.index - datetime(1970, 1, 1)
    ).total_seconds()  # total seconds elapsed of each data point since 1970
    vol_perc = df[ts_input_name]
    smooth = lowess(vol_perc, df["time_sec"], is_sorted=True, frac=0.01, it=0)
    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "time_stamp"})
    df = pd.merge(df, df_smooth, on="time_sec")  # merge smooth signal into origianl dataframe
    df.set_index("time_stamp", drop=True, append=False, inplace=True, verify_integrity=False)

    # STEP 3: Create new time series
    if data["dry_run"]:
        ts_output = client.time_series.create(TimeSeries(name=ts_output_name, external_id=ts_output_name))
    else:
        ts_output = "hei"
        ts_output = client.time_series.create(
            TimeSeries(name=ts_output_name, external_id=ts_output_name, data_set_id=data_set_id)
        )

    # STEP 4: Calculate daily average drainage rate
    df["derivative"] = np.gradient(df["smooth"], df["time_sec"])  # Unit: vol_percentage/time [% of tank vol / sec]
    # replace when derivative is greater than alfa
    derivative_value_excl = data["derivative_value_excl"]
    df["derivative_excl_filling"] = df["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x
    )

    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["time_stamp"]).dt.date
    mean_drainage_day = (
        df.groupby("Date")["derivative_excl_filling"].mean() * tank_volume / 100
    )  # avg drainage rate per DAY

    mean_df = pd.DataFrame({ts_output_name: mean_drainage_day})  # Use external ID as column name

    new_df = pd.merge(df, mean_df, on="Date")
    new_df["draining_rate [L/min]"] = (
        new_df["derivative_excl_filling"] * tank_volume / 100
    )  # drainage rate per TIME STAMP

    mean_df.index = pd.to_datetime(mean_df.index)
    ts_inserted = client.time_series.data.insert_dataframe(mean_df)
    # ts_inserted = "hei"

    if data["dry_run"]:  # Delete ts if testing locally
        client.time_series.delete(external_id=ts_output_name)
        print(f"ts_output: {ts_output}")
        print(f"ts_inserted: {ts_inserted}")

    return new_df[[ts_output_name]].to_json()  # , ts_output, ts_inserted
