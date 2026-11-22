"""Load the public data for the FOMC factor project.

french_daily(path): first block (value-weighted returns) of a Ken French daily CSV
                    inside its zip, as a DataFrame in percent, date index, -99.99 as NaN.
portfolio_panel(dir): the 384-series panel used in the preliminary tables
                    (100 size-B/M, 100 size-OP, 100 size-INV, 25 size-momentum,
                    10 momentum, 49 industries), in excess of the daily T-bill rate.
fomc_days(csv):     scheduled FOMC announcement days and all other Fed policy events.

"""
import io
import os
import zipfile

import numpy as np
import pandas as pd

PANEL_FILES = [
    ("BM", "100_Portfolios_10x10_Daily_CSV.zip"),
    ("OP", "100_Portfolios_ME_OP_10x10_daily_CSV.zip"),
    ("INV", "100_Portfolios_ME_INV_10x10_daily_CSV.zip"),
    ("SzMom", "25_Portfolios_ME_Prior_12_2_Daily_CSV.zip"),
    ("Mom", "10_Portfolios_Prior_12_2_Daily_CSV.zip"),
    ("Ind", "49_Industry_Portfolios_daily_CSV.zip"),
]


def french_daily(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        name = z.namelist()[0]
        lines = z.read(name).decode("latin-1").splitlines()
    # header = first line that starts with a comma (column names); data until a blank line
    start = next(i for i, l in enumerate(lines) if l.lstrip().startswith(","))  # international files start with " ,"
    block = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip() or not l.strip()[0].isdigit():
            break
        block.append(l)
    df = pd.read_csv(io.StringIO("\n".join(block)), skipinitialspace=True)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str).str.strip(), format="%Y%m%d")
    df = df.set_index("date").astype(float)
    df[df <= -99.99] = np.nan
    return df


def portfolio_panel(french_dir, start="1994-01-01", end=None):
    ff = french_daily(os.path.join(french_dir, "F-F_Research_Data_Factors_daily_CSV.zip"))
    parts = []
    for tag, fname in PANEL_FILES:
        df = french_daily(os.path.join(french_dir, fname))
        df.columns = [f"{tag}:{c}" for c in df.columns]
        parts.append(df)
    X = pd.concat(parts, axis=1)
    X = X.loc[start:end] if end else X.loc[start:]
    X = X.sub(ff.loc[X.index, "RF"], axis=0)          # excess returns, percent per day
    return X, ff.loc[X.index]


def fomc_days(meetings_csv):
    m = pd.read_csv(meetings_csv, parse_dates=["date"])
    scheduled = pd.DatetimeIndex(m.loc[m.kind == "scheduled", "date"]).sort_values()
    other = pd.DatetimeIndex(m.loc[m.kind.isin(
        ["unscheduled", "conference_call", "notation_vote", "cancelled", "other_meeting"]), "date"])
    return scheduled, other.sort_values()
