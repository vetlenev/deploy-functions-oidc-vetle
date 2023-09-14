import datetime

from datetime import timedelta

from cognite.client.data_classes import TimeSeries
from prophet import Prophet

from common.utilities import thermal_resistance


def thermal_resistance_forecast(df):
    """Function to forecast the Thermal Resistance"""
    df2 = df.copy()[["TR"]].reset_index()
    df2 = df2.rename(columns={"index": "ds", "TR": "y"})
    m = Prophet()
    m.fit(df2)
    future = m.make_future_dataframe(periods=24 * 15, freq="H")
    future["cap"] = 1.1 * df["TR"].mean()  #
    fcst = m.predict(future)
    fcst_df = fcst[["ds", "yhat"]].set_index("ds")
    fcst_df.columns = ["TR"]
    return fcst_df


def create_and_save_time_series_data(client, data, ts_external_id, data_set_id):
    """Function to create the time series and save the data"""
    cdf_ts = client.time_series.retrieve(external_id=ts_external_id)
    if cdf_ts is None:
        ts = TimeSeries(external_id=ts_external_id, name=ts_external_id, unit="m2K/W", data_set_id=data_set_id)
        client.time_series.create(ts)
        print("Created time series")
    else:
        print("Existing Time Series")

    dps = []
    for index, r in data.iterrows():
        dps = dps + [{"timestamp": r.name, "value": r["TR"]}]
        client.datapoints.insert(datapoints=dps, external_id=ts_external_id)


def handle(client, data=None, secrets=None, function_call_info=None):
    """Handler Function to be Run/Deployed
    Args:
        client : Cognite Client (not needed, it's availble to it, when deployed)
        data : data needed by function
        secrets : Any secrets it needs
        function_call_info : any other information about function

    Returns:
        response : response or result from the function
    """
    data_set_id = client.data_sets.retrieve(external_id="functions-playground").id
    ts_exids = ["pi:163657", "pi:163658", "pi:160887", "pi:191092", "pi:163374", "pi:160184"]
    column_names = ["T_cold_IN", "T_cold_OUT", "T_hot_IN", "T_hot_OUT", "Flow_cold", "Flow_hot"]
    your_name = data["your_name"]
    # Retrieve the data
    start_date = datetime.datetime(2018, 8, 1)
    end_date = start_date + timedelta(days=30)
    df = client.datapoints.retrieve_dataframe(
        external_id=ts_exids,
        aggregates=["average"],
        granularity="1h",
        start=start_date,
        end=end_date,
        include_aggregate_name=False,
    )
    df.fillna(method="ffill", inplace=True)
    df.columns = column_names
    # Calculate the Thermal resistance
    df["TR"] = df.apply(lambda x: thermal_resistance(x), axis=1)
    # Forecast the Thermal resistance
    fcst_df = thermal_resistance_forecast(df)
    # Save the Results as time series
    create_and_save_time_series_data(client, df[["TR"]], f"hx_thermal_resistance_{your_name}", data_set_id=data_set_id)
    create_and_save_time_series_data(
        client, fcst_df[["TR"]], f"hx_thermal_resistance_forecast_{your_name}", data_set_id=data_set_id
    )
    # Return the result as json
    result = df[["TR"]].to_json()
    return result
