def modulation_series(params: dict, limit: int = 26):
    """Return matching plot coordinates for an IntervalsModulations config.

    The launcher represents N change times with N+1 values: the extra first
    value applies before the first change. Some fixtures use one timestamp per
    value, so retain support for that representation as well.
    """
    times = list(params["interval_times"])
    values = list(params["interval_values"])
    if len(values) == len(times) + 1:
        times.insert(0, 0)
    elif len(values) != len(times):
        raise ValueError(
            "interval_values must contain either one value per interval_time "
            "or one additional initial value; "
            f"got {len(values)} values and {len(times)} times"
        )
    return times[:limit], values[:limit]
