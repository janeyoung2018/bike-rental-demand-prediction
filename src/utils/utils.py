def create_xgb_features(data):
    df_feat = data.copy()
    # add day in a month as feature
    df_feat["day"] = df_feat["datetime"].dt.day
    # lag and rolling statistics
    df_feat["lag_1"] = df_feat["cnt"].shift(1)
    df_feat["rolling_24"] = df_feat["cnt"].shift(1).rolling(24).mean()
    df_feat["rolling_168"] = df_feat["cnt"].shift(1).rolling(168).mean()
    # statistics
    aggregations = ["mean", "max", "min", "std", "sum"]
    # Extract the date part just once
    df_feat["date_only"] = df_feat["dteday"].dt.date
    # Apply each aggregation using a loop
    for agg in aggregations:
        df_feat[f"day_{agg}"] = (
            df_feat.groupby("date_only")["cnt"].transform(agg).shift(24)
        )
    return df_feat
