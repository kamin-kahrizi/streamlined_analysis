import os.path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import pandas as pd
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess
from scipy.signal import savgol_filter as savgol
import os
from pathlib import Path
import shutil
import re
import seaborn as sns
import json
import numbers

from datetime import datetime
import time
from raw_data import RawData
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

matplotlib.use("Agg")

import sqlite3
from jinja2 import Environment, BaseLoader
import webbrowser
import tempfile
import plotly.express as px
import plotly.graph_objects as go
import itertools
import functools
from scipy.stats import linregress


def template():
    return """
    <body>
        <div>        
        <p style="border: 2px solid; background-color:yellow"> Excluded sensors: {{excluded_sensors}} </p>
        </div>
        {% for fig, name, csv_str in figs %}
        <h1> {{ name }} </h1>
        {% if csv_str is not none %}
           <a download="{{name}}.csv", href="data:text/csv;charset=utf-8, {{csv_str|safe}}"> Download {{name}} </a>
           {% endif %}

           {{ fig|safe }}
        {% endfor %}
        <details>
        <summary> Metadata </summary>
        {% for fname, data in metadata %}
          <a download="{{fname}}", href="data:text;charset=utf-8, {{data|d('', true)|e}}">{{fname}} </a>
        {% endfor %}
        </details>
    </body>
    """


def match_axis_by_row(fig, kind):
    assert kind in ["x", "y"], "Kind must be x or y"
    layout = fig.to_plotly_json()["layout"]

    axis_names = [(k, v) for k, v in layout.items() if f"{kind}axis" in k]

    yval = lambda x: x[1]["domain"][0]
    for k, g in itertools.groupby(sorted(axis_names, key=yval), key=yval):
        axes = list(g)
        match_name = axes[0][0].replace("axis", "")
        for k, _ in axes:
            fig.update_layout({k: {"matches": match_name}})


def flow_figure(window, dir_name, reads, values):
    df = processed_summaries(window, dir_name, reads, values)
    df.sort_values("Time", inplace=True)
    vals = ["Peak WL", "centroid", "fwhm", "Peak I"]
    # average over replicates
    group_vars = ["Read group", "Condition", "Sample ID"]
    df_avgd = df.groupby(group_vars)[vals + ["Time"]].mean(numeric_only=False)

    rg_baseline = df.sort_values("Time")["Read group"].iloc[0]

    # process differences
    df_diffs = df_avgd - df_avgd.loc[rg_baseline]
    df_diffs.reset_index(inplace=True)
    df_diffs["Conc."] = df_diffs["Read group"].apply(lambda x: x.split("_")[0])
    df_diffs["Time (min)"] = df_diffs["Time"].apply(lambda x: x.total_seconds() / 60)
    df_diffs["Read number"] = df_diffs["Time"].rank(method="dense")

    df_melted = df_diffs.melt(
        id_vars=(set(df_diffs.columns) - set(vals)), value_vars=vals
    )
    df_melted.sort_values(by="Time (min)", inplace=True)

    # process raw values
    df_raw = df_avgd.reset_index()
    df_raw["Conc."] = df_raw["Read group"].apply(lambda x: x.split("_")[0])
    df_melted_raw = df_raw.melt(
        id_vars=set(df_raw.columns) - set(vals), value_vars=vals
    )
    df_melted_raw["Read number"] = df_melted_raw["Time"].rank(method="dense")
    df_melted_raw.sort_values(by="Time", inplace=True)

    def plot_x(x_var, df):
        fig = px.line(
            df.reset_index(),
            x=x_var,
            y="value",
            color="Conc.",
            facet_col="Condition",
            facet_row="variable",
            line_group="Sample ID",
            category_orders={"variable": vals},
            markers=True,
        )
        match_axis_by_row(fig, "y")
        return fig

    # plot diffs
    fig_time = plot_x("Time (min)", df_melted)
    fig_read_num = plot_x("Read number", df_melted)

    add_toggle(fig_time, "Time", fig_read_num, "Read number", "x")

    # plot raw
    fig_time_raw = plot_x("Time", df_melted_raw)
    fig_read_num_raw = plot_x("Read number", df_melted_raw)
    add_toggle(fig_time_raw, "Time", fig_read_num_raw, "Read number", "x")
    plotly_page(
        [
            (fig_time, "Differences", df_diffs.to_csv()),
            (fig_time_raw, "Raw", df_raw.to_csv()),
        ],
        get_html_metadata(window, dir_name, values),
    )


@functools.cache
def get_time(fname):
    timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
    timestamp = timestamp_pat.search(fname).group(2)
    return datetime.strptime(timestamp, "%Y_%m_%d_%H_%M_%S")


def processed_summaries(window, dir_name, reads, values):
    # load summaries (without excluded)
    df = filtered_summaries(dir_name, reads, window, values)

    # get sensor key
    try:
        df_key = window["-SENSOR LEGEND-"].metadata["legend"]
        condition = lambda smp: df_key.loc[smp, "Label"]
    except:
        condition = lambda x: x

    cols = [
        "Sample ID",
        "Peak WL",
        "Peak I",
        "f_sg",
        "I_max_smp",
        "fwhm",
        "centroid",
        "File",
    ]

    df = df[cols]
    df["Read group"] = df["File"].apply(read_group)
    df["Time"] = df["File"].apply(get_time)
    df["Condition"] = df["Sample ID"].apply(condition)

    return df


def strip_box_multiplot(df_melted):
    fig_all_box = px.box(
        df_melted,
        x="Condition",
        y="value",
        color="Read group",
        facet_col="metric",
        facet_col_wrap=2,
    )
    fig_all_strip = px.strip(
        df_melted,
        x="Condition",
        y="value",
        color="Read group",
        facet_col="metric",
        facet_col_wrap=2,
        hover_data=["Sample ID", "value"],
    )

    fig_all_both = go.Figure()
    fig_all_both.add_traces([*fig_all_box["data"], *fig_all_strip["data"]])
    fig_all_both.update_layout(fig_all_strip.layout)
    fig_all_both.update_layout()
    fig_all_both.update_yaxes(matches=None, showticklabels=True)
    return fig_all_both


def condition_plot(window, dir_name, reads, values):
    df = processed_summaries(window, dir_name, reads, values)
    df.sort_values("Time", inplace=True)
    vals = ["Peak WL", "centroid", "fwhm", "Peak I"]
    # average over replicates
    group_vars = ["Read group", "Condition", "Sample ID"]
    df_avgd = df.groupby(group_vars)[vals].mean()

    rg_baseline = df.sort_values("Time")["Read group"].iloc[0]

    df_diffs = df_avgd - df_avgd.loc[rg_baseline]

    diffs_melted = pd.melt(
        df_diffs.reset_index(), id_vars=group_vars, value_vars=vals, var_name="metric"
    )
    raw_melted = pd.melt(
        df_avgd.reset_index(), id_vars=group_vars, value_vars=vals, var_name="metric"
    )

    fig_diffs = strip_box_multiplot(diffs_melted)
    fig_raw = strip_box_multiplot(raw_melted)
    plotly_page(
        [
            (fig_diffs, "Differences", df_diffs.to_csv()),
            (fig_raw, "Raw", df_avgd.to_csv()),
        ],
        get_html_metadata(window, dir_name, values),
    )


def get_analysis_params(dir_name):
    params_path = Path(dir_name, "parameters.json")

    with open(params_path, "r") as f:
        return json.dumps(json.load(f))


def get_key(window):
    if window["-SENSOR LEGEND-"].metadata is not None:
        return window["-SENSOR LEGEND-"].metadata["legend"].to_csv()

    return None


def get_html_metadata(window, dir_name, values):
    return {
        "analyis_params.json": get_analysis_params(dir_name),
        "key.csv": get_key(window),
        "plot_params.json": json.dumps(values),
    }


def plotly_page(figs, metadata):
    excluded_sensors = json.loads(metadata["plot_params.json"])["-OMIT SENSORS-"]
    figs_html = []
    for i, (fig, name, csv_str) in enumerate(figs):
        if i == 0:
            js = "cdn"
        else:
            js = False
        figs_html.append(
            (fig.to_html(full_html=False, include_plotlyjs=js), name, csv_str)
        )
    temp = Environment(loader=BaseLoader).from_string(template())
    html_str = temp.render(
        figs=figs_html, metadata=metadata.items(), excluded_sensors=excluded_sensors
    )

    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".html") as f:
        url = "file://" + f.name
        f.write(html_str)
    webbrowser.open(url)


def f_time(name):
    timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
    get_time = lambda x: datetime.strptime(
        timestamp_pat.search(x).group(2), "%Y_%m_%d_%H_%M_%S"
    )
    if isinstance(name, pd.Series):
        return [get_time(x) for x in name]
    else:
        return get_time(name)


def read_summaries(dir_name, reads):
    df = pd.DataFrame()
    for read in reads:
        sum_data = pd.read_csv(
            "{}/{}_analysis/{}_summary.csv".format(dir_name, read, read), comment="#"
        )
        sum_data["File"] = read
        sum_data["Sample ID"] = sum_data["Sample ID"].apply(lambda x: int(x.split()[1]))

        df = pd.concat([df, sum_data], ignore_index=True)

    return df


def filtered_summaries(dir_name, reads, window, values):
    peak_df = read_summaries(dir_name, reads)
    smps = get_sensors(window, dir_name, reads, values)
    peak_df = peak_df.query("`Sample ID` in @smps")
    return peak_df


def get_metadata(dir_name, filtered_reads):
    # get metadata
    # check if all files have metadata and that they match
    metadatas = []
    for read in filtered_reads:
        metadata_path = Path(dir_name, f"{read}_analysis", "metadata.json")

        if metadata_path.is_file():
            with open(metadata_path, "r") as f:
                curr_metadata = json.load(f)
                metadatas.append(curr_metadata)
        else:
            metadatas.append(None)

    if all([x == metadatas[0] for x in metadatas]):
        metadata = metadatas[0]
    else:
        metadata = None

    return metadata


def add_toggle(fig1, button_label1, fig2, button_label2, kind):
    assert kind in ["x", "y"], "Kind must be x or y"

    label1 = fig1.layout[f"{kind}axis"].title.text
    label2 = fig2.layout[f"{kind}axis"].title.text

    # get yaxes with labels for relabeling with toggle
    layout = fig1.layout.to_plotly_json()
    axis_names = [k for k, v in layout.items() if (f"{kind}axis" in k) and "title" in v]

    def axis_dict(new_label):
        return {f"{x}.title.text": new_label for x in axis_names}

    fig1.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            {kind: [trace[kind] for trace in fig1.data]},
                            axis_dict(label1),
                        ],
                        label=button_label1,
                        method="update",
                    ),
                    dict(
                        args=[
                            {kind: [trace[kind] for trace in fig2.data]},
                            axis_dict(label2),
                        ],
                        label=button_label2,
                        method="update",
                    ),
                ]
            )
        ]
    )


def baseline(spec_df, lo, hi, window, extra_group=[]):
    def bkg_subtract(df):
        inds = (
            (df.Wavelength < (lo + window / 2)) & (df.Wavelength > (lo - window / 2))
        ) | ((df.Wavelength < (hi + window / 2)) & (df.Wavelength > (hi - window / 2)))

        res = linregress(df[inds].Wavelength, df[inds].Absorbance)
        bl = res.intercept + res.slope * df.Wavelength

        return pd.DataFrame(
            {"wl": df.Wavelength, "abs_corr": df.Absorbance - bl}
        ).set_index("wl")

    return (
        spec_df.groupby(
            ["sample", "read group", "condition"] + extra_group, group_keys=True
        )
        .apply(bkg_subtract)
        .reset_index()
    )


def spectra_plot(df, y_name="abs_corr"):
    wls = sorted(df.wl.unique())[::10]
    return px.line(
        df.query("wl in @wls"),
        x="wl",
        y=y_name,
        facet_col="condition",
        facet_col_wrap=8,
        color="read group",
        line_group="sample",
    )


def spectra_figure_bl_corr(dir_name, reads, values, window):
    t_start = datetime.now()
    cols = [
        "Wavelength",
        "Absorbance Fit (unnormalized)",
        "read",
        "sample",
        "read group",
        "condition",
    ]

    df_spec = collect_spectra(dir_name, reads, values, window, cols=cols)

    print(datetime.now() - t_start)

    rg_baseline = read_group(sorted(df_spec["read"].unique(), key=get_time)[0])

    # making dataframe less bulky
    df_spec["time"] = df_spec["read"].apply(get_time)
    df_spec.drop(columns=["read"], inplace=True)
    df_spec["read group"] = df_spec["read group"].astype("category")
    df_spec["condition"] = df_spec["condition"].astype("category")

    df_all = df_spec.rename(columns = 
                            {'Absorbance Fit (unnormalized)': 'Absorbance'})
    print(f'pre grouping processing: {datetime.now() - t_start}')

    df_spec = (df_spec.groupby(
        ['read group', 'sample', 'Wavelength', 'condition'], observed = True)
               ['Absorbance Fit (unnormalized)'].mean()).rename('Absorbance')
    print(f'grouping: {datetime.now() - t_start}')

    fig_raw = spectra_plot(
        pd.DataFrame(df_spec).reset_index().rename(columns={"Wavelength": "wl"}),
        y_name="Absorbance",
    )

    print(f'plot fig_raw: {datetime.now() - t_start}')
    df_diff = (df_spec - df_spec[rg_baseline])

    df_diff_all = (
        df_all.set_index(["time", "read group", "sample", "Wavelength", "condition"])[
            "Absorbance"
        ]
        - df_spec[rg_baseline]
    )

    fig_diff = spectra_plot(
        pd.DataFrame(df_diff).reset_index().rename(columns={"Wavelength": "wl"}),
        y_name="Absorbance",
    )

    print(f'plot fig_diff: {datetime.now() - t_start}')
    wl_bl_lo = 620
    wl_bl_hi = 935
    bl_window = 10

    df_bl = baseline(pd.DataFrame(df_diff).reset_index(), wl_bl_lo, wl_bl_hi, bl_window)

    df_bl_all = baseline(
        pd.DataFrame(df_diff_all).reset_index(),
        wl_bl_lo,
        wl_bl_hi,
        bl_window,
        extra_group=["time"],
    )

    fig_bl = spectra_plot(df_bl)

    print(f'plot fig_bl: {datetime.now() - t_start}')
    # condition plot
    def i_max_diff(spec_df, extra_group=[]):
        group_cols = ["sample", "read group", "condition"] + extra_group
        df_pk = pd.DataFrame(
            spec_df.query("770 < wl < 800").groupby(group_cols)["abs_corr"].max()
        )
        df_shoulder = pd.DataFrame(
            spec_df.query("670 < wl < 690").groupby(group_cols)["abs_corr"].max()
        )
        return (df_pk - df_shoulder).reset_index()

    peak_abs = i_max_diff(df_bl).sort_values("read group").dropna()
    peak_abs_all = i_max_diff(df_bl_all, extra_group=["time"]).dropna()

    cond_order = df_all.groupby("read group").time.min().sort_values().index.to_list()

    fig_peak_abs_strip = px.strip(
        peak_abs,
        x="condition",
        y="abs_corr",
        color="read group",
        hover_data=["sample"],
        category_orders={"read group": cond_order},
    )
    fig_peak_abs_box = px.box(
        peak_abs,
        x="condition",
        y="abs_corr",
        color="read group",
        hover_data=["sample"],
        category_orders={"read group": cond_order},
    )

    fig_peak_abs = go.Figure()
    fig_peak_abs.add_traces([*fig_peak_abs_box["data"], *fig_peak_abs_strip["data"]])
    fig_peak_abs.update_layout(fig_peak_abs_strip.layout)
    fig_peak_abs.update_layout()
    fig_peak_abs.update_yaxes(matches=None, showticklabels=True)

    fig_peak_abs_time = px.line(
        peak_abs_all,
        x="time",
        y="abs_corr",
        color="read group",
        hover_data=["sample"],
        facet_col="condition",
        facet_col_wrap = 10,
        line_group="sample",
        markers=True,
    )

    # ---- code to get rid of the "condition=" in the plots ----
    for fig in [fig_raw, fig_diff, fig_bl, fig_peak_abs, fig_peak_abs_time]:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    plotly_page(
        [
            (fig_raw, "Raw Spectra", None),
            (fig_diff, "Difference Spectra", None),
            (fig_bl, "Baselined spectra", None),
            (fig_peak_abs, "Condition Plot", peak_abs.to_csv(escapechar="\\")),
            (fig_peak_abs_time, "Time course", peak_abs_all.to_csv(escapechar="\\")),
        ],
        get_html_metadata(window, dir_name, values),
    )


def spectra_figure(dir_name, reads, values, window):
    t_start = datetime.now()
    cols = [
        "Wavelength",
        "Absorbance Fit (unnormalized)",
        "sensor",
        "read",
        "read_index",
        "sample",
        "read group",
        "condition",
    ]

    df_spec = collect_spectra(dir_name, reads, values, window, cols=cols)
    print(datetime.now() - t_start)

    rg_baseline = read_group(sorted(df_spec["read"].unique(), key=get_time)[0])

    df_spec = df_spec[cols]
    df_spec = (
        df_spec.groupby(["read group", "sample", "Wavelength", "condition"])[
            "Absorbance Fit (unnormalized)"
        ].mean()
    ).rename("Absorbance")

    df_norm = (
        df_spec / df_spec.groupby(["read group", "sample", "condition"]).max()
    ).rename("Absorbance (norm.)")

    def plot_spec_series(series):
        fig = px.line(
            pd.DataFrame(series).reset_index(),
            x="Wavelength",
            y=series.name,
            facet_col="condition",
            facet_col_wrap=5,
            color="read group",
            line_group="sample",
            render_mode="svg",
        )

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        return fig

    # full spectra plot
    fig_raw = plot_spec_series(df_spec)
    fig_norm = plot_spec_series(df_norm)
    add_toggle(fig_raw, "Unnormalized", fig_norm, "Normalized", "y")

    # difference spectra plot
    df_diff = (df_spec - df_spec[rg_baseline]).rename("Abs-Abs(0)")
    df_diff_norm = (df_norm - df_norm[rg_baseline]).rename("Abs norm. - Abs. norm(0)")

    fig_diff = plot_spec_series(df_diff)
    fig_diff_norm = plot_spec_series(df_diff_norm)
    add_toggle(fig_diff, "Unnormalized", fig_diff_norm, "Normalized", "y")

    plotly_page(
        [(fig_raw, "Raw Spectra", None), (fig_diff, "Difference Spectra", None)],
        get_html_metadata(window, dir_name, values),
    )


def collect_spectra(dir_name, reads, values, window, cols=None):
    # smps = get_sensors(window, dir_name, reads, values)
    # metadata = get_metadata(dir_name, reads)

    summary_df = filtered_summaries(dir_name, reads, window, values)

    if cols is None:
        cols = [
            "Wavelength",
            "Absorbance Fit (unnormalized)",
            "sensor",
            "read",
            "read_index",
            "sample",
            "read group",
        ]

    else:
        cols = [x for x in cols if x != "condition"]

    # collect spectral data
    spec_dfs = []
    # smps_by_read = {read:number_of_samples(dir_name, read) for read in reads}

    for read_index, read in enumerate(sorted(reads, key=f_time)):
        grp = read_group(read)

        read_smps = summary_df.query("File == @read")["Sample ID"]

        sql_path = Path(dir_name, f"{read}_analysis", f"{read}.sqlite")

        if sql_path.exists():
            con = sqlite3.connect(sql_path)

        for i in read_smps:
            csv_path = Path(dir_name, f"{read}_analysis", f"{read}_sample_{i}.csv")
            hdf_path = Path(dir_name, f"{read}_analysis", f"{read}.hdf5")

            if csv_path.exists():
                data = pd.read_csv(csv_path)
            elif hdf_path.exists():
                data = pd.read_hdf(hdf_path, str(i))
            elif sql_path.exists():
                data = pd.read_sql(f"SELECT * from spectra where sensor=={i}", con)

            data["read"] = read
            data["read_index"] = read_index
            data["sample"] = i

            data["read group"] = read_group(read)
            spec_dfs.append(data[cols])

        if sql_path.exists():
            con.close()
    spec_df = pd.concat(spec_dfs)

    # get sensor key
    try:
        df_key = window["-SENSOR LEGEND-"].metadata["legend"]
        condition = lambda smp: df_key.loc[smp, "Label"]
    except:
        condition = lambda x: x

    spec_df["condition"] = spec_df["sample"].apply(condition)
    return spec_df


def read_group(read):
    pat_ts = re.compile(r"(_[A-Z])*_[0-9]{4}(_[0-9]{2}){5}$")
    return pat_ts.sub("", read)


def number_of_samples(dir_name, read):
    df = pd.read_csv(f"{dir_name}/{read}_analysis/{read}_summary.csv", comment="#")
    return len(pd.unique(df["Sample ID"]))


def get_sensors(window, dir_name, reads, values):
    summaries = read_summaries(dir_name, reads)
    sens_omit_select = window["-OMIT SENSORS-"]
    omit_sens = values["-OMIT SENSORS-"]
    smps_df = set(summaries["Sample ID"])
    return smps_df - set(omit_sens)


def update_sensor_list(window, dir_name, reads, values):
    peak_df = read_summaries(dir_name, reads)
    sens_omit_select = window["-OMIT SENSORS-"]
    smps_df = set(peak_df["Sample ID"])
    curr_vals = values["-OMIT SENSORS-"]

    sens_omit_select.update(smps_df)
    sens_omit_select.set_value(curr_vals)


def plot_transmittance(ax, wl, bg_raw, sg_raw, peak_WL, ctd, wl_range, bg_higher, t0):
    ax.plot(wl, bg_raw, label="background raw", color="gray")
    ax.plot(wl, sg_raw, label="signal", color="lightcoral")
    ax.vlines(peak_WL, -2, 2, color="g", linestyles="dashed")
    ax.vlines(ctd, -2, 2, color="orange", linestyles="dashed")
    ax.vlines(wl_range, [-2, -2], [2, 2], linestyles="dotted", color="gray")
    ax.set_ylim([np.amin(bg_raw) - 0.01, np.amax(bg_raw) + 0.01])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (au)")
    ax.legend(["bkg", "sig"])

    at1 = AnchoredText(
        r"$f_{sg}$ = %0.1f" % (t0), prop=dict(size=10), frameon=True, loc="upper left"
    )
    at1.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at1)


def plot_absorbance(
    ax,
    wl,
    wl_range,
    bg_raw,
    sg_raw,
    peak_WL,
    abs,
    abs_sm,
    Steepness,
    fwhm,
    t0,
    noise,
    bg_higher,
    ctd,
):
    """

    Args:
        wl (_type_): _description_
        wl_range (_type_): _description_
        bg_raw (_type_): _description_
        sg_raw (_type_): _description_
        sample_no (_type_): _description_
        peak_WL (_type_): _description_
        abs (_type_): _description_
        abs_sm (_type_): _description_
        Steepness (_type_): _description_
        Q_factor (_type_): _description_
        noise (_type_): _description_
        bg_higher (_type_): _description_
        filename (_type_): _description_
        savedir (_type_): _description_
    """

    ax.vlines(peak_WL, -2, 2, color="g", label="peak", linestyles="dashed")
    ax.vlines(ctd, -2, 2, color="orange", label="centroid", linestyles="dashed")
    ax.plot(wl, abs, label="Absorption", color="lightblue")
    ax.plot(wl, abs_sm, label="Loess fit", color="k")
    ax.legend(["peak", "centroid"], loc="upper right")
    ax.set_ylim([np.amin(abs_sm) - 0.01, np.amax(abs_sm) + 0.3])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Absorption (au)")
    ax.vlines(wl_range, [-2, -2], [2, 2], linestyles="dotted", color="gray")

    at = AnchoredText(
        "$λ_{p}$ = %0.2f nm \n$λ_{c}$ = %0.2f nm\n$\sigma$ = %0.2f $x 1e^{-2}$\n$A_{max}$ = %0.3f\nFWHM = %.0f nm"
        % (peak_WL, ctd, 100 * noise, max(abs_sm), fwhm),
        prop=dict(size=10),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)

    return ax


def DataAnalysisM1(
    Bg_raw, Sg_raw, WL, WL_range, WL_range_FWHM, WL_range_Amax, Smooth, Norm_Abs
):

    BgSg = np.union1d(Bg_raw, Sg_raw)
    minval = np.amin(BgSg)

    # Find resolution of intensitiy
    res = np.amin(np.amin(np.abs(BgSg[np.nonzero(BgSg)])))

    # Set offset
    bg_offset = Bg_raw - minval + res
    sg_offset = Sg_raw - minval + res

    # Normalize Background and Signal intensities such that the peak is == 1

    Bg = bg_offset / np.amax(bg_offset)
    Sg = sg_offset / np.amax(sg_offset)

    # Find intensitiy depreication between Background and Signal
    T0 = np.amax(sg_offset) / np.amax(bg_offset)

    # Calculate Absorbance
    WL_roi_ind = np.where((WL > WL_range[0]) & (WL < WL_range[1]))
    WL_roi = WL[WL_roi_ind]
    Abs = -np.log10(Sg / Bg)

    Abs_no_trans = -np.log10(sg_offset / bg_offset)
    Abs_no_norm = Abs
    Abs_loess = lowess(Abs, WL, frac=Smooth / 100.0)
    Abs_sm = Abs_loess[:, 1]

    # shift spectrum so that minimum absorbance is 0
    amax_inds = (WL > WL_range_Amax[0]) & (WL < WL_range_Amax[1])
    abs_shift = np.min(Abs_sm[amax_inds])
    Abs_sm = Abs_sm - abs_shift
    Abs = Abs - abs_shift
    Abs_no_norm = Abs_no_norm - abs_shift

    min_Norm = np.amin(Abs_sm[WL_roi_ind])
    peak_Norm = np.amax(Abs_sm[WL_roi_ind] - min_Norm)
    Abs_norm = (Abs - min_Norm) / peak_Norm
    Abs_sm_no_norm = Abs_sm

    if Norm_Abs:
        Abs = Abs_norm
        Abs_sm = (Abs_sm - min_Norm) / peak_Norm

    Abs_sm_norm = (Abs_sm_no_norm - min_Norm) / peak_Norm

    # Data smoothing using loess
    # Alternatively can use savgol

    Abs_roi = Abs_sm[WL_roi_ind]

    peak_ind = (np.diff(np.sign(np.diff(Abs_roi))) < 0).nonzero()[0] + 1

    if len(peak_ind) == 0:
        peak_WL = np.nan
        Q = np.nan
    # incase multiplpe peaks are found, pick the the peak with the highest intensity value

    else:
        peak_WL = WL_roi[peak_ind][
            np.where(Abs_roi[peak_ind] == np.amax(Abs_roi[peak_ind]))
        ][0]

    peak_H = np.ptp(Abs_sm[amax_inds])

    if np.isnan(peak_WL):
        Q = np.nan
        steepness = np.nan
        sigma_noise = np.nan

    else:
        sigma_noise = np.std(Abs[WL_roi_ind] - Abs_roi)
        steepness = -np.mean(np.diff(np.diff(Abs_roi)))
        Q = T0 * (peak_H / sigma_noise) * steepness * 1950000

    fwhm_inds = (WL >= WL_range_FWHM[0]) & (WL <= WL_range_FWHM[1])
    # calculate fwhm of smoothed spectrum
    hm = np.max(Abs_sm[fwhm_inds]) - (np.ptp(Abs_sm[fwhm_inds]) / 2)
    overall_peak = peak_WL
    wl_hi = overall_peak
    wl_lo = overall_peak

    for wl in sorted(WL[WL >= overall_peak]):
        if Abs_sm[WL == wl] >= hm:
            wl_hi = wl
        else:
            break
    for wl in sorted(WL[WL < overall_peak], reverse=True):
        if Abs_sm[WL == wl] >= hm:
            wl_lo = wl
        else:
            break

    fwhm = wl_hi - wl_lo
    if not isinstance(fwhm, numbers.Number):
        fwhm = 0

    #### CENTROID MEASUREMENTS #####
    Abs_ctd = Abs_roi - np.amin(Abs_roi)
    centrd = np.sum(Abs_ctd * WL_roi) / np.sum(Abs_ctd)

    # Find average signal & background and flag is signal<background
    Sg_avg = np.mean(Sg_raw)
    Bg_avg = np.mean(Bg_raw)

    if Bg_avg > Sg_avg:
        Bg_higher = "True"
    else:
        Bg_higher = "False"

    return (
        Bg,
        Sg,
        bg_offset,
        sg_offset,
        Abs,
        Abs_sm,
        T0,
        peak_WL,
        peak_H,
        Q,
        steepness,
        sigma_noise,
        centrd,
        Bg_higher,
        Abs_no_trans,
        Abs_no_norm,
        Abs_norm,
        fwhm,
        Abs_sm_norm,
        Abs_sm_no_norm,
    )


def Analyze_file(
    run_name, f, wl_range, wl_range_fwhm, wl_range_amax, Smooth, Norm_Abs, version
):
    fname = Path(f).stem
    dir_name = os.path.dirname(f)

    # if Save_flag:
    analysis_dir = Path(dir_name, f"{fname}_analysis")
    if not os.path.isdir(analysis_dir):
        os.mkdir(analysis_dir)

    shutil.copy2(f, analysis_dir)

    raw_data = RawData(f)

    # get legend information (sample metadata)
    locs = raw_data.read_locations()
    meta_cols = ["On Target", "Conjugated PNA"]
    sample_metadata = {}
    if set(meta_cols) <= set(locs.columns):
        for i in locs.index:
            sample_metadata[i] = locs[meta_cols].loc[i].to_dict()

    # get spectra
    wl = raw_data.wl

    df_summary = pd.DataFrame(
        columns=[
            "Sample ID",
            "Bkgd>Sample?",
            "Peak WL",
            "Peak I",
            "Noise",
            "Steepness",
            "f_sg",
            "I_max_smp",
            "Q",
            "fwhm",
            "centroid",
            "Wavel Range",
            "Smoothing",
            "Absorbance Normalized",
        ]
    )

    con = sqlite3.connect("{}/{}.sqlite".format(analysis_dir, fname))

    for d in raw_data.spectra():
        sample_ID = "sample {}".format(str(d["sensor"]))
        sg_raw = d["spectrum"]
        bg_raw = d["background"]

        (
            Bg,
            Sg,
            bg_offset,
            sg_offset,
            Abs,
            Abs_sm,
            T0,
            peak_WL,
            peak_H,
            Q,
            steepness,
            sigma_noise,
            ctd,
            Bg_higher,
            Abs_no_trans,
            Abs_no_norm,
            Abs_norm,
            fwhm,
            Abs_sm_norm,
            Abs_sm_no_norm,
        ) = DataAnalysisM1(
            bg_raw, sg_raw, wl, wl_range, wl_range_fwhm, wl_range_amax, Smooth, Norm_Abs
        )

        Processed_Data = pd.DataFrame(
            columns=["Wavelength", "Absorbance Fit (unnormalized)"]
        )
        Processed_Data["Wavelength"] = wl

        Processed_Data["Absorbance Fit (unnormalized)"] = Abs_sm_no_norm

        # save to SQL
        Processed_Data["sensor"] = d["sensor"]
        Processed_Data.to_sql("spectra", con, if_exists="append", index=False)
        # Processed_Data.to_parquet(f"{fname}.gzip", compression='gzip')

        # Save to CSV
        # Processed_Data.to_csv('{}/{}_sample_{}.csv'.format(analysis_dir,fname,d['sensor']))

        # Save to HD5
        # Processed_Data.astype('float32').to_hdf('{}/{}.hdf5'.format(analysis_dir,fname), d['sensor'])
        if d["sensor"] == "1":
            WLRANGE = "-".join(str(i) for i in wl_range)
            SMOOTH = str(Smooth)
            if Norm_Abs:
                ABSNRM = "YES"
            else:
                ABSNRM = "NO"
        else:
            WLRANGE = ""
            SMOOTH = ""
            ABSNRM = ""

        record = dict(
            zip(
                df_summary.columns,
                [
                    sample_ID,
                    Bg_higher,
                    peak_WL,
                    peak_H,
                    sigma_noise,
                    steepness,
                    T0,
                    max(sg_offset),
                    Q,
                    fwhm,
                    ctd,
                    WLRANGE,
                    SMOOTH,
                    ABSNRM,
                ],
            )
        )

        df_summary = pd.concat([df_summary, pd.DataFrame([record])])

    # close spectra SQLite connection
    con.close()

    df_summary = df_summary.sort_values(by="Sample ID")

    fname_summary = "{}/{}_summary.csv".format(analysis_dir, fname)
    with open(fname_summary, "w") as f:
        f.write("# NpDataAnalysis v{}\n".format(version))

    df_summary.to_csv(fname_summary, mode="a")

    metadata_path = Path(analysis_dir, "metadata.json")

    if sample_metadata != {}:
        with open(metadata_path, "w") as f:
            json.dump(sample_metadata, f)

    return analysis_dir  ####################3


# Function that checks the max intensity in the raw data file
def check_max_saturation(df_temp, filename, popup_msg):
    df_temp.columns = [col.strip() for col in df_temp.columns]
    sample_cols = [col for col in df_temp.columns if "Sample" in str(col)]

    df_temp[sample_cols] = df_temp[sample_cols].apply(pd.to_numeric, errors="coerce")

    saturation_mask = df_temp[sample_cols] > 64000

    saturated_rows = saturation_mask.any(axis=1)

    for idx in df_temp.index[saturated_rows]:
        saturated_cols = saturation_mask.loc[idx][
            saturation_mask.loc[idx]
        ].index.tolist()

        saturated_info = [f"{col}: {df_temp.loc[idx, col]}" for col in saturated_cols]
        message = f"File: {filename}\nRow: {idx}\n" + "\n".join(saturated_info)
        popup_msg.append(message)

    return popup_msg
