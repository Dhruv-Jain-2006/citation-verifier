def classify_quota_error(e: Exception) -> str:
    err_str = str(e).lower()

    # Strong DAILY quota indicators (hard stop)
    rpd_signals = [
        "perday",
        "requests per day",
        "generaterequestsperday",
        "daily quota",
        "exceeded your current quota",
        "free_tier_requests"
    ]

    # Strong MINUTE quota indicators (retryable)
    rpm_signals = [
        "perminute",
        "requests per minute",
        "retry_delay",
        "please retry"
    ]

    # Check RPM first (to avoid false breaker)
    for signal in rpm_signals:
        if signal in err_str:
            return "RPM"

    # Then check RPD
    for signal in rpd_signals:
        if signal in err_str:
            return "RPD"

    return "UNKNOWN"