import pandas as pd
from pandas.io.excel import ExcelWriter
from traffic_observer.sector_manager import SectorManager
import logging

def create_stats_report(sector_cluster: SectorManager, report_path: str):
    traffic_stats = sector_cluster.traffic_stats()
    classwise_stats = sector_cluster.classwise_stats()
    logging.info("Созданы датафреймы со статистикой.")

    res_dataframes = []
    i = 1
    for traf_stat, class_stat in zip(traffic_stats, classwise_stats):
        df_res_tmp = pd.concat([traf_stat, class_stat], axis=1)
        res_dataframes.append(df_res_tmp)

        print("*********************")
        print(f"Sector #{i}")
        print(df_res_tmp)
        i += 1

    # Запись данных в файл
    with ExcelWriter(report_path) as writer:
        for ind, df in enumerate(res_dataframes):
            df.to_excel(writer, sheet_name=f"{ind + 1}")