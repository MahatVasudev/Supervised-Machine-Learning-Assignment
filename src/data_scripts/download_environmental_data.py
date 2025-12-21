import cdsapi

dataset = "derived-era5-single-levels-daily-statistics"
request = {
    "product_type": "ensemble_mean",
    "variable": ["10m_u_component_of_wind"],
    "year": "2018",
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",
    "frequency": "3_hourly"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
