# Paper input panel

`df_union_sqrt.csv` is the eight-market panel used by the current manuscript:
3,531 union-calendar dates, 2013-03-05 through 2026-09-28, with 2,870 dates
on which all markets are active. It replaces the repository's earlier
2006-2022 dataset; the filename is retained for compatibility with the
existing directory layout.

Columns, in order: `.SPX`, `.GDAXI`, `.FCHI`, `.FTSE`, `.OMXSPI`, `.N225`,
`.KS11`, `.HSI`. The first column is the date index.

Values are the daily Parkinson volatility proxy in percent:

```text
100 * log(high / low) / sqrt(4 * log(2))
```

Use `--input-format vol_pct`, the default. These are daily range-based
volatilities, not five-minute realized volatility or aggregated h-day targets.
Missing entries retain asynchronous market activity. Do not forward-fill
the target panel or replace absent observations with zero volatility.

The source is Yahoo Finance historical daily index observations. The
archived supporting OHLC snapshot was retrieved on 2026-09-30; that timestamp
is not asserted as an independently recorded download time for the training
panel. The supporting OHLC values were checked against the panel proxy.
Raw OHLC files are not included in this release.

The exact CSV SHA-256 is:

```text
8d13af29b4534ba614ccce7bd7bf0bd24b29627ef3d280873a45792912143bb6
```

Third-party data are outside the code's MIT license; see `../NOTICE.md`.
