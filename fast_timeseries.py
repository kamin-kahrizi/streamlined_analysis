import argparse
from pathlib import Path
import re
from datetime import datetime

import numpy as np
import pandas as pd

from raw_data import RawData
import offsetters
import referencers
import smoothers
from scipy.stats import linregress


def read_group(read: str) -> str:
    """Strip trailing timestamp and optional suffixes from filename to get read group."""
    pat_ts = re.compile(r"(_[A-Z])*_[0-9]{4}(_[0-9]{2}){5}$")
    return pat_ts.sub("", read)


def get_time(read: str) -> datetime:
    timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
    timestamp = timestamp_pat.search(read).group(2)
    return datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")


def baseline(spec_df: pd.DataFrame, lo: float, hi: float, window: float, extra_group=None) -> pd.DataFrame:
    """Baseline correct spectra using linear regression at windowed regions."""
    if extra_group is None:
        extra_group = []

    def bkg_subtract(df):
        inds = ((df.Wavelength < (lo + window / 2)) & (df.Wavelength > (lo - window / 2))) | (
            (df.Wavelength < (hi + window / 2)) & (df.Wavelength > (hi - window / 2))
        )
        res = linregress(df[inds].Wavelength, df[inds].Absorbance)
        bl = res.intercept + res.slope * df.Wavelength
        return pd.DataFrame({"wl": df.Wavelength, "abs_corr": df.Absorbance - bl}).set_index("wl")

    group_cols = ["sample", "read group", "condition"] + list(extra_group)
    return (
        spec_df.groupby(group_cols, group_keys=True)
        .apply(bkg_subtract)
        .reset_index()
    )


def i_max_diff(spec_df: pd.DataFrame, extra_group=None) -> pd.DataFrame:
    """Difference between peak and shoulder regions of baseline corrected spectra."""
    if extra_group is None:
        extra_group = []
    group_cols = ["sample", "read group", "condition"] + list(extra_group)
    df_pk = pd.DataFrame(
        spec_df.query("770 < wl < 800").groupby(group_cols)["abs_corr"].max()
    )
    df_shoulder = pd.DataFrame(
        spec_df.query("670 < wl < 690").groupby(group_cols)["abs_corr"].max()
    )
    return (df_pk - df_shoulder).reset_index()


def process_folder(folder: Path, legend: Path | None, output: Path) -> None:
    files = sorted(folder.glob('*.csv'))
    if not files:
        raise FileNotFoundError('No data files found')

    spectra_frames = []
    legend_map = {}
    if legend is not None:
        df_key = pd.read_csv(legend)
        if {'Sensor', 'Label'}.issubset(df_key.columns):
            legend_map = dict(zip(df_key['Sensor'], df_key['Label']))
        elif {'#', 'Label'}.issubset(df_key.columns):
            legend_map = dict(zip(df_key['#'], df_key['Label']))

    for f in files:
        read = f.stem
        rg = read_group(read)
        t = get_time(read)
        r = RawData(f)
        wl = r.wl
        for spec in r.spectra():
            data = {'wavelength': wl, 'spectra': {'bkg': spec['background'], 'smp': spec['spectrum']}}
            for fn in (
                offsetters.global_offset,
                referencers.ln_abs,
                smoothers.loess_smoother(0.0125),
            ):
                data = fn(data)
            abs_spec = data['spectra']['abs']
            spectra_frames.append(pd.DataFrame({
                'Wavelength': wl,
                'Absorbance': abs_spec,
                'sample': int(spec['sensor']),
                'read group': rg,
                'time': t,
            }))

    df_all = pd.concat(spectra_frames, ignore_index=True)
    df_all['condition'] = df_all['sample'].map(legend_map).fillna(df_all['sample'])

    baseline_group = (
        df_all.groupby('read group')['time'].min().sort_values().index[0]
    )
    baseline_spec = (
        df_all[df_all['read group'] == baseline_group]
        .groupby(['sample', 'Wavelength'])['Absorbance']
        .mean()
        .rename('baseline')
    )
    df_diff = df_all.merge(baseline_spec, on=['sample', 'Wavelength'], how='left')
    df_diff['Absorbance'] = df_diff['Absorbance'] - df_diff['baseline']
    df_diff.drop(columns='baseline', inplace=True)

    df_bl = baseline(df_diff, 620, 935, 10, extra_group=['time'])
    timeseries_df = i_max_diff(df_bl, extra_group=['time'])
    timeseries_df.to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description='Generate baseline corrected time series from data folder')
    parser.add_argument('folder', type=Path, help='Folder containing raw data files')
    parser.add_argument('-o', '--output', type=Path, default=Path('timeseries.csv'), help='Output CSV file')
    parser.add_argument('--legend', type=Path, help='Optional sensor legend CSV')
    args = parser.parse_args()
    process_folder(args.folder, args.legend, args.output)


if __name__ == '__main__':
    main()
