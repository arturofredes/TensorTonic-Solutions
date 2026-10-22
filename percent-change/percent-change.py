def percent_change(series: list) -> list:
    """
    Returns the fractional change between consecutive values.
    """
    # Write code here
    p_change = []
    for i in range(1,len(series)):
        prev = series[i - 1]
        if prev == 0:
            p_change.append(0)
        else:
            p_change.append((series[i] - prev) / prev)
    
    return p_change    