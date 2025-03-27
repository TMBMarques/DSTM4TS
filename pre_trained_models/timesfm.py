import timesfm

def run_timesfm(df, horizon_length):

    # Initialize the model and load a checkpoint

    tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          horizon_len=horizon_length,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )

    """ tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
        context_len=168,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
    ) """

    """ tfm = timesfm.TimesFm(
        context_len=168,
        horizon_len=24,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=20,
        model_dims=1280,
        backend="cpu",
    )
    tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m") """

    """ tfm = timesfm.TimesFm(
      hparams=timesfm.TimesFmHparams(
          backend="cpu",
          per_core_batch_size=32,
          context_len=168,
          horizon_len=24,
      ),
      checkpoint=timesfm.TimesFmCheckpoint(
          huggingface_repo_id="google/timesfm-1.0-200m"),
  ) """


    # Load base class

    """ Note that the four parameters are fixed to load the 200m model """
    """ input_patch_len=32
    output_patch_len=128
    num_layers=20
    model_dims=1280

    tcm = TimesFMConfig(input_patch_len, output_patch_len, num_layers, model_dims)
    tfm_torch = PatchedTimeSeriesDecoder(tcm)
    tfm_torch.load_state_dict(torch.load(PATH)) """

    """ model handles a max context length of 512 """
    """ tfm.forecast(<the input time series contexts>, <frequency>) """

    """ 
        frequency 0 -> up to daily granularity (T, MIN, H, D, B, U)
        frequency 1 -> weekly and monthly granularity (W, M)
        frequency 2 -> anything beyond monthly granularity, e.g. quarterly or yearly (Q, Y) 
    """
    
    
    # Forecast

    # numpy
    """ forecast_input = [
        np.sin(np.linspace(0, 20, 100)),
        np.sin(np.linspace(0, 20, 200)),
        np.sin(np.linspace(0, 20, 400)),
    ]
    frequency_input = [0, 1, 2]
    point_forecast, experimental_quantile_forecast = tfm.forecast(
        forecast_input,
        freq=frequency_input,
    ) """

    # pandas
    forecast_df = tfm.forecast_on_df(
        inputs=df,
        freq="h",  # hourly
        value_name="y",
        num_jobs=-1,
    )

    forecast = forecast_df.tail(24)
    forecast = forecast.iloc[:, :3]
    #forecast.to_csv('forecast_output.txt', index=False, sep='\t')

    return forecast
