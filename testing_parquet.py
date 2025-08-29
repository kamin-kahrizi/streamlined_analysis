import pandas as pd
from pathlib import Path
from raw_data import RawData
import os
import sqlite3
import parquet
import time
import pyarrow as pa
import pyarrow.parquet as pq
"""
load sqlite file into a df and then load into parquet file and save that. compare the sizes of that


want to now:
    create a folder that will hold the multiple sqlite files in it
    - have a loop that writes the data to sqlite a lot and see how long it takes
    - change that 
"""

# conn = sqlite3.connect("test_data/Wash 1_rep1_2025_01_27_11_24_25.sqlite")
# df = pd.read_sql_query("SELECT * FROM spectra", conn)
# print(type(df))
# print(f"sqlite: \n {df}")
# df_parquet = df.to_parquet('df_parquet.gzip', compression='gzip')

# df_parquet_read = pd.read_parquet('df_parquet.gzip')

# print(f"parquet:\n {df_parquet_read}")
# conn.close()


def run_sqlite(tracker):
    folder = Path("sqlite Folder")
    folder.mkdir(exist_ok=True)

    source_sqlite_path = Path("data") / "air_reads_overnight_2-AA_2025_01_23_18_28_16.sqlite"
    source_conn = sqlite3.connect(source_sqlite_path)
    
    df_temp = pd.read_sql_query("SELECT * FROM spectra", source_conn)
    source_conn.close()

    dest_sqlite_path = folder/f"{source_sqlite_path.stem}_copy{tracker}.sqlite"

    dest_conn = sqlite3.connect(dest_sqlite_path)

    df_temp.to_sql("spectra", dest_conn, if_exists="replace", index=False)
    dest_conn.close()

def run_parquet(tracker):
    folder = Path("parquet Folder")
    folder.mkdir(exist_ok=True)

    source_sqlite_path = Path("data") / "air_reads_overnight_2-AA_2025_01_23_18_28_16.sqlite"

    source_conn = sqlite3.connect(source_sqlite_path)

    df = pd.read_sql_query("SELECT * FROM spectra", source_conn)
    source_conn.close()

    dest_sqlite_path = folder/f"{source_sqlite_path.stem}_copy{tracker}.gzip"

    df.to_parquet(dest_sqlite_path, compression='gzip')
    
    # df_parquet_read = pd.read_parquet('df_parquet.gzip')

    # print(f"parquet:\n {df_parquet_read}")


def read_write_parquet(tracker):
    #want to read the parquet file from data folder
    folder = Path("parquet Folder")

    source_sqlite_path = Path("data") / "air_reads_overnight_2-AA_2025_01_23_18_28_16.gzip"
    #extract the df from parquet
    df_parquet_read = pd.read_parquet(source_sqlite_path)

    dest_folder = Path("parquet Folder")

    dest_path = dest_folder / f"air_reads_overnight_2-AA_2025_01_23_18_28_16_TEST3{tracker}.gzip"

    #write to parquet file and put into parquet folder
    df_parquet_read.to_parquet(dest_path, compression='gzip', engine='pyarrow')
    

tracker = 0
iterations = 26
start_time = time.perf_counter()
for i in range(iterations):
    # run_sqlite(tracker)
    # run_parquet(tracker)
    read_write_parquet(tracker)
    tracker+=1

end_time = time.perf_counter()

total_time = end_time - start_time
print(f"ran function {iterations} times in {total_time:.2f} secs") #ran 44.04: 1st time


    
