if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()

import FreeSimpleGUI as sg
import os.path
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

import traceback
import pandas as pd
import numpy as np
import statsmodels.api as sm

lowess = sm.nonparametric.lowess
from scipy.signal import savgol_filter as savgol
import os
from pathlib import Path

from NpDatFuncs_v0_2 import *
from raw_data import RawData

from time import time
from datetime import datetime
version = '0.11.1'

from time import perf_counter
import argparse
from pyinstrument import Profiler
import cam_concerned
profiler = Profiler()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-bm",
    "--benchmarking",
    action="store_true",
    help="Runs a testbench to see how long the NPDA runs",
)
args = parser.parse_args()

cam_concered_image = cam_concerned.img

def main():
    plot_spectra = [
        sg.Button(
            "Plot spectra", disabled=True, key="-PLOT SPECTRA-", enable_events=True
        )
    ]
    plot_spectra_bl = [
        sg.Button(
            "Plot spectra (BL)",
            disabled=True,
            key="-PLOT SPECTRA BL-",
            enable_events=True,
        )
    ]
    plot_box = [
        sg.Button(
            "Plot by condition",
            disabled=True,
            key="-PLOT CONDITION-",
            enable_events=True,
        )
    ]
    plot_flow = [
        sg.Button(
            "Plot by condition (flow)",
            disabled=True,
            key="-PLOT FLOW-",
            enable_events=True,
        )
    ]

    # First the window layout in 2 columns
    sg.theme("Purple")
    file_list_column = [
        [
            sg.Text("Select Raw Data File"),
            sg.In(size=(25, 1), enable_events=False, key="-RAW_DATA-"),
            sg.FilesBrowse(),
        ],
        [
            sg.Text("Min WL (nm)    "),
            sg.InputText("700", size=(10, 1), enable_events=False, key="-MIN_WL-"),
            sg.Text("Max WL (nm)   "),
            sg.InputText("900", size=(10, 1), enable_events=False, key="-MAX_WL-"),
        ],
        [
            sg.Text("Min WL (FWHM) (nm)    "),
            sg.InputText("550", size=(10, 1), enable_events=False, key="-MIN_WL_FWHM-"),
            sg.Text("Max WL (FHWM) (nm)   "),
            sg.InputText(
                "1050", size=(10, 1), enable_events=False, key="-MAX_WL_FWHM-"
            ),
        ],
        [
            sg.Text("Min WL (A_max) (nm)    "),
            sg.InputText("620", size=(10, 1), enable_events=False, key="-MIN_WL_Amax-"),
            sg.Text("Max WL (A_max) (nm)   "),
            sg.InputText("920", size=(10, 1), enable_events=False, key="-MAX_WL_Amax-"),
        ],
        [
            sg.Text("Smoothing (%) "),
            sg.In("4", size=(10, 1), enable_events=False, key="-SMOOTH-"),
        ],
        [
            sg.Button("Analyze", key="-ANALYZE-"),
        ],
        [
            sg.Text("Select processed files for overlay"),
            sg.In(
                size=(10, 1), key="-OVERLAY FILES-", enable_events=True, visible=False
            ),
            sg.FilesBrowse(enable_events=True, target="-OVERLAY FILES-"),
        ],
        [
            sg.Listbox(
                values=[],
                select_mode="extended",
                enable_events=True,
                size=(35, 10),
                key="-READ LIST-",
            ),
            sg.Listbox(
                values=[], select_mode="extended", size=(5, 10), key="-OMIT SENSORS-"
            ),
            sg.Column(
                [
                    [
                        sg.Button("Exclude fsg <", key="-AUTO EXCLUDE-"),
                        sg.InputText(
                            "0.8", size=(5, 1), enable_events=False, key="-FSG-"
                        ),
                    ]
                ]
            ),
        ],
        [
            sg.FileBrowse(
                "Load sensor legend", enable_events=True, key="-SENSOR LEGEND-"
            )],
        [
            sg.Column([plot_spectra, plot_spectra_bl]),
            sg.Column([plot_box]),
            sg.Column([plot_flow]),
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose legends to plot specfic \nconditions:", key="-LEGEND TEXT-", visible=False)],
        [sg.Text(size=(25, 1), key="-TOUT-")],
        [sg.Image(key="-IMAGE-")],
        [sg.Canvas(key="-CANVAS-")],
        [sg.Canvas(key="-CANVAS 2-")],

        [sg.Listbox(values = [],
                   select_mode='extended',
                   enable_events=True,
                    size=(20, 10),
                    key="-READ LIST CONDITIONS PLOT-",
                    visible=False
                        )]
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    window = sg.Window(
        "Nanopath Data Analysis Tool v{}".format(version),
        layout,
        finalize=True,
        resizable=True,
    )

    if not hasattr(window, 'processed_keys'):
        window.processed_keys = set()

    # Run the Event Loop
    while True:
        try:
            event, values = window.read()
            print(event)
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            # Folder name was filled in, make a list of files in the folder

            if event == "-ANALYZE-":
                start_test = perf_counter()
                print("-ANALYZE- time testing start")

                file = values["-RAW_DATA-"].split(";")

                min_wl = int(values["-MIN_WL-"])
                max_wl = int(values["-MAX_WL-"])
                min_wl_fwhm = int(values["-MIN_WL_FWHM-"])
                max_wl_fwhm = int(values["-MAX_WL_FWHM-"])
                min_wl_amax = int(values["-MIN_WL_Amax-"])
                max_wl_amax = int(values["-MAX_WL_Amax-"])

                smooth = float(values["-SMOOTH-"])
                norm_flag = False

                cdt1 = (min_wl > 400) & (max_wl < 1101) & (min_wl < max_wl)
                ctd2 = smooth < 100

                if cdt1 & ctd2:
                    # generate analysis run folder
                    run_name = datetime.now().strftime("%Y_%m_%d %H%M%S_analysis")
                    parent_dir = Path(file[0]).parent
                    analysis_dir = Path(parent_dir, run_name)
                    os.mkdir(analysis_dir)

                    with open(Path(analysis_dir, "parameters.json"), "w") as f_params:
                        json.dump(values, f_params)

                    for f in file:
                        shutil.copy2(f, analysis_dir)

                    file = [Path(analysis_dir, Path(f).name) for f in file]

                    # === Reads each file into a dataframe and add file name ===
                    dfs = []
                    popup_messages = []
                    for f in file:
                        file_name = f.name
                        if f.suffix.lower() == ".csv":
                            object_data = RawData(f)
                            df_temp = object_data.read_spectra()
                            # print(df_temp)
                        else:
                            sg.popup_annoying(f"Unsupported file type: {f.suffix}")
                            continue

                        df_temp["File Name"] = file_name
                        dfs.append(df_temp)

                        # --- Checking for saturation ---
                        popup_messages = check_max_saturation(
                            df_temp, file_name, popup_messages
                        )

                    if len(file) == 1:
                        try:
                            save_dir = Analyze_file(
                                run_name,
                                file[0],
                                (min_wl, max_wl),
                                (min_wl_fwhm, max_wl_fwhm),
                                (min_wl_amax, max_wl_amax),
                                smooth,
                                norm_flag,
                                version,
                            )

                            # --- code to display the single file in the '-READ LIST-' textbox ---#
                            window["-READ LIST-"].metadata = {
                                "dir": Path(file[0]).parent
                            }

                            df = pd.DataFrame()
                            fname_base = Path(file[0]).stem
                            df_curr = pd.read_csv(
                                "{}/{}_summary.csv".format(save_dir, fname_base),
                                comment="#",
                            )
                            df_curr["File name"] = fname_base
                            df = pd.concat([df, df_curr])

                            parent_dir = Path(file[0]).parent
                            first_cols = [
                                "File name",
                                "Unnamed: 0",
                                "Sample ID",
                                "Peak WL",
                                "Q",
                                "centroid",
                            ]
                            cols = [
                                x for x in df.columns.tolist() if x not in first_cols
                            ]

                            cols = first_cols + cols
                            fname_full_summary = "{}/full_summary.csv".format(
                                parent_dir
                            )
                            with open(fname_full_summary, "w") as f:
                                f.write("# NpDataAnalysis v{}\n".format(version))
                            df[cols].to_csv(fname_full_summary, mode="a")

                            reads = [Path(f).stem for f in file]
                            timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
                            f_time = lambda x: datetime.strptime(
                                timestamp_pat.search(x).group(2), "%Y_%m_%d_%H_%M_%S"
                            )
                            window["-READ LIST-"].update(sorted(reads, key=f_time))

                            # autosave_plots(reads, parent_dir)
                            # print(time() - strt)

                            # --- TIME TESTING ---
                            end_test = perf_counter()

                            print(
                                f"\n-ANALYZE- took {end_test - start_test:.6f} seconds to finsih.\n"
                            )

                        except Exception as e:
                            print(traceback.format_exc())
                            print(e)
                            sg.popup_annoying("something went wrong")

                    # handle multiple files
                    else:
                        n_cpu = mp.cpu_count()
                        pool = mp.Pool(processes=int(0.75 * n_cpu))

                        save_dirs = list(
                            pool.starmap(
                                Analyze_file,
                                zip(
                                    [run_name] * len(file),
                                    file,
                                    [(min_wl, max_wl)] * len(file),
                                    [(min_wl_fwhm, max_wl_fwhm)] * len(file),
                                    [(min_wl_amax, max_wl_amax)] * len(file),
                                    [smooth] * len(file),
                                    [norm_flag] * len(file),
                                    [version] * len(file),
                                ),
                            )
                        )

                        pool.close()

                        # store directory name in READ LIST metadata
                        window["-READ LIST-"].metadata = {"dir": Path(file[0]).parent}

                        # collect all summaries into single file
                        df = pd.DataFrame()
                        for i in range(0, len(save_dirs)):
                            fname_base = Path(file[i]).stem
                            df_curr = pd.read_csv(
                                "{}/{}_summary.csv".format(save_dirs[i], fname_base),
                                comment="#",
                            )
                            df_curr["File name"] = fname_base
                            df = pd.concat([df, df_curr])

                        parent_dir = Path(file[0]).parent
                        first_cols = [
                            "File name",
                            "Unnamed: 0",
                            "Sample ID",
                            "Peak WL",
                            "Q",
                            "centroid",
                        ]
                        cols = [x for x in df.columns.tolist() if x not in first_cols]

                        cols = first_cols + cols
                        fname_full_summary = "{}/full_summary.csv".format(parent_dir)
                        with open(fname_full_summary, "w") as f:
                            f.write("# NpDataAnalysis v{}\n".format(version))
                        df[cols].to_csv(fname_full_summary, mode="a")

                        reads = [Path(f).stem for f in file]
                        timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
                        f_time = lambda x: datetime.strptime(
                            timestamp_pat.search(x).group(2), "%Y_%m_%d_%H_%M_%S"
                        )
                        window["-READ LIST-"].update(sorted(reads, key=f_time))

                        # autosave_plots(reads, parent_dir)
                        # print(time() - strt)

                        # --- TIME TESTING ---
                        end_test = perf_counter()

                        print(
                            f"\n-ANALYZE- took {end_test - start_test:.6f} seconds to finsih.\n"
                        )

                    #printing out saturation samples after analyse is done
                    if popup_messages:
                        final_msg = "\n\n".join(popup_messages)
                        sg.popup_scrolled(
                            "Saturated Samples Found:",
                            final_msg,
                            image=cam_concered_image
                        )

                else:
                    sg.popup_annoying("Incompatible Analysis Parameters")
                    window.refresh()

            if event == "-SENSOR LEGEND-": #and "-SENSOR LEGEND-" not in window.processed_keys:
                window.processed_keys.add("-SENSOR LEGEND-") 

                df = pd.read_excel(values["-SENSOR LEGEND-"]).set_index("Sensor")
                window["-SENSOR LEGEND-"].metadata = {"legend": df}
                
                sensor_labels = list(df['Label'].unique())

                window["-LEGEND TEXT-"].update(visible=True)

                window["-READ LIST CONDITIONS PLOT-"].update(values=sensor_labels, visible=True)


            if event == "-MIN_WL-":
                # this is necessary for autoplotting, see the manual additions to the
                # event queue at the end of the 'Analyze' event
                window["-MIN_WL-"].update(value=values["-MIN_WL-"])
                window.processed_keys.add("-MIN_WL-")

            if event == "-MAX_WL-":
                # this is necessary for autoplotting, see the manual additions to the
                # event queue at the end of the 'Analyze' event
                window["-MAX_WL-"].update(value=values["-MAX_WL-"])
                window.processed_keys.add("-MAX_WL-")

            if event == "-OVERLAY FILES-":
                overlay_files = values["-OVERLAY FILES-"].split(";")
                parent_dir = Path(overlay_files[0]).parent
                window["-READ LIST-"].metadata = {"dir": parent_dir}
                window["-OMIT SENSORS-"].metadata = None

                fnames = [Path(f).stem for f in overlay_files]
                timestamp_pat = re.compile(r"(.*)([0-9]{4}(_[0-9]{2}){5})")
                f_time = lambda x: datetime.strptime(
                    timestamp_pat.search(x).group(2), "%Y_%m_%d_%H_%M_%S"
                )
                window["-READ LIST-"].update(sorted(fnames, key=f_time))

                window.processed_keys.add("-OVERLAY FILES-")
                
            if event == "-AUTO EXCLUDE-":
                sens = (
                    read_summaries(
                        window["-READ LIST-"].metadata["dir"], values["-READ LIST-"]
                    )
                    .query("f_sg < @fsg_cut")["Sample ID"]
                    .unique()
                )
                window["-OMIT SENSORS-"].set_value(sens)
                window.processed_keys.add("-AUTO EXCLUDE-")
                
            if event == "-READ LIST CONDITIONS PLOT-":
                selected_labels = values["-READ LIST CONDITIONS PLOT-"]
                window['-SENSOR LEGEND-'].metadata["selected_sensors"] = selected_labels

                legend_df = window["-SENSOR LEGEND-"].metadata["legend"]

                selected_sensor_ids = legend_df[legend_df['Label'].isin(selected_labels)].index.tolist()

                sensor_box_values = window["-OMIT SENSORS-"].get_list_values()
                indices_to_select = [
                    i for i, sensor_id in enumerate(sensor_box_values)
                    if sensor_id not in selected_sensor_ids
                ]

                window["-OMIT SENSORS-"].update(set_to_index=indices_to_select)
                window.processed_keys.add("-READ LIST CONDITIONS PLOT-")
                

            if event == "-READ LIST-":
                start_plot_test = perf_counter()

                # add transmission plots to -FILE LIST- if only
                # one file is selected

                window["-PLOT SPECTRA-"].update(disabled=False)
                window["-PLOT SPECTRA BL-"].update(disabled=False)
                window["-PLOT CONDITION-"].update(disabled=False)
                window["-PLOT FLOW-"].update(disabled=False)

                update_sensor_list(
                    window,
                    window["-READ LIST-"].metadata["dir"],
                    values["-READ LIST-"],
                    values,
                )

                reads = values["-READ LIST-"]
                window.processed_keys.add("-READ LIST-")

            # select read for plotting/overlaying by sensor
            if event in [
                "-PLOT SPECTRA-",
                "-PLOT SPECTRA BL-",
                "-PLOT CONDITION-",
                "-PLOT FLOW-",
            ]:
                vals = values["-READ LIST-"]
                if len(vals) == 0:
                    continue
                # generate overlay figure
                match event:
                    case "-PLOT SPECTRA-":
                        spectra_figure(
                            window["-READ LIST-"].metadata["dir"],
                            values["-READ LIST-"],
                            values,
                            window,
                        )

                        end_plot_test = perf_counter()
                        print(
                            f"\nPlot test took {end_plot_test-start_plot_test:.6f} seconds"
                        )

                    case "-PLOT SPECTRA BL-":
                        spectra_figure_bl_corr(
                            window["-READ LIST-"].metadata["dir"],
                            values["-READ LIST-"],
                            values,
                            window,
                        )
                        end_plot_test = perf_counter()
                        print(
                            f"\nPlot test took {end_plot_test-start_plot_test:.6f} seconds"
                        )

                    case "-PLOT CONDITION-":
                        condition_plot(
                            window,
                            window["-READ LIST-"].metadata["dir"],
                            values["-READ LIST-"],
                            values,
                        )
                        end_plot_test = perf_counter()
                        print(
                            f"\nPlot test took {end_plot_test-start_plot_test:.6f} seconds"
                        )

                    case "-PLOT FLOW-":
                        flow_figure(
                            window,
                            window["-READ LIST-"].metadata["dir"],
                            values["-READ LIST-"],
                            values,
                        )
                        end_plot_test = perf_counter()
                        print(
                            f"\nPlot test took {end_plot_test-start_plot_test:.6f} seconds"
                        )

        except Exception as e:
            sg.popup_error("Error", traceback.format_exc(), e, image=cam_concered_image)
            pass
    window.close()


if __name__ == "__main__":
    if args.benchmarking:
        profiler.start()
        main()
        profiler.stop()
        print("Benchmarking results:")
        print(profiler.output_text())
    else:
        # start = perf_counter()
        main()
        # end = perf_counter()
        # print(f"Elapsed time: {end - start:.6f} seconds")
