import pandas as pd
import numpy as np
from collections import defaultdict
from traffic_observer.crossroad_manager import DataCollector

def create_excel_report(datacollector: DataCollector, report_path: str):
    data_dict = defaultdict(lambda: defaultdict(list))
    
    for vehicle in datacollector.collected_vehicles:
        # Only include vehicles with a defined end_id
        if vehicle.end_id is not None:
            data_dict[vehicle.lane_id][vehicle.end_id].append((vehicle.start_delay, vehicle.travel_time))
    
    # Get sorted unique lane and end values
    lanes = sorted(data_dict.keys())
    ends = set()
    for lane in data_dict:
        ends.update(data_dict[lane].keys())
    ends = sorted(ends)
    
    # Prepare a dictionary for DataFrame construction.
    # Each key is a tuple (end, metric) and each value is the list of averages for the corresponding lane.
    df_data = {}
    for end in ends:
        avg_start_delay = []
        avg_travel_time = []
        for lane in lanes:
            values = data_dict[lane].get(end)
            if values:
                # Compute averages if records exist.
                delays = [v[0] for v in values]
                times = [v[1] for v in values]
                avg_delay = sum(delays) / len(delays)
                avg_time = sum(times) / len(times)
            else:
                avg_delay = np.nan
                avg_time = np.nan
            avg_start_delay.append(avg_delay)
            avg_travel_time.append(avg_time)
        df_data[(end, "start_delay")] = avg_start_delay
        df_data[(end, "travel_time")] = avg_travel_time

    # Create a MultiIndex for columns. Top level are the end_ids and second level are the metrics.
    multi_columns = pd.MultiIndex.from_tuples(list(df_data.keys()), names=["end", ""])
    
    # Create DataFrame from the prepared dictionary.
    df = pd.DataFrame(df_data, index=lanes)
    df.index.name = "lane"
    df = df.reindex(columns=multi_columns)
    
    # Write the DataFrame to an Excel file.
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Report")
    
    print(f"Excel report generated: {report_path}")
    return df