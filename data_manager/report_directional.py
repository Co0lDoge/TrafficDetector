import pandas as pd
import os
import logging
from traffic_observer.crossroad_manager import DataCollector

def create_report_dataframe(datacollector: DataCollector):
    from collections import defaultdict
    import pandas as pd
    import numpy as np

    # Group data by (direction_id, lane_id) and then by end_id.
    # Each vehicle contributes a tuple: (start_delay, travel_time)
    data_dict = defaultdict(lambda: defaultdict(list))
    for vehicle in datacollector.collected_vehicles:
        if vehicle.end_id is not None:
            # Use a tuple key representing the start direction and lane.
            key = (vehicle.direction_id, vehicle.lane_id)
            data_dict[key][vehicle.end_id].append((vehicle.start_delay, vehicle.travel_time))
    
    # Get a sorted list of the unique (direction_id, lane_id) keys.
    index_keys = sorted(data_dict.keys())
    
    # Determine sorted unique end ids from all groups.
    ends = set()
    for key in data_dict:
        ends.update(data_dict[key].keys())
    ends = sorted(ends)
    
    # Prepare a dictionary that will be used to build the DataFrame.
    # For each end, calculate:
    #  - the average start_delay (only including values >= 10)
    #  - the average travel_time (using all values)
    #  - the vehicle count
    df_data = {}
    for end in ends:
        avg_start_delay = []
        avg_travel_time = []
        vehicle_count = []
        for key in index_keys:
            values = data_dict[key].get(end)
            if values:
                # Filter start_delay values less than 10.
                valid_delays = [v[0] for v in values if v[0] >= 10]
                avg_delay = sum(valid_delays) / len(valid_delays) if valid_delays else np.nan
                # For travel_time, include all values.
                times = [v[1] for v in values]
                avg_time = sum(times) / len(times)
                count = len(values)
            else:
                avg_delay = np.nan
                avg_time = np.nan
                count = np.nan
            avg_start_delay.append(avg_delay)
            avg_travel_time.append(avg_time)
            vehicle_count.append(count)
            
        # For each end, store its three metrics.
        df_data[(end, "start_delay")] = avg_start_delay
        df_data[(end, "travel_time")] = avg_travel_time
        df_data[(end, "vehicle_count")] = vehicle_count

    # Create a MultiIndex for the columns: top level will be end_id and the second level will be the metric.
    multi_columns = pd.MultiIndex.from_tuples(list(df_data.keys()), names=["end", ""])
    
    # Create a MultiIndex for the DataFrame rows from the (start_direction, lane) tuples.
    index = pd.MultiIndex.from_tuples(index_keys, names=["start_direction", "lane"])
    
    # Construct the DataFrame and reindex using the constructed MultiIndexes.
    df = pd.DataFrame(df_data, index=index)
    df = df.reindex(columns=multi_columns)

    return df

def create_report(datacollector: DataCollector, report_path: str):
    _, ext = os.path.splitext(report_path)
    ext = ext.lower()

    if ext == ".json":
        return create_json_report(datacollector, report_path)
    elif ext == ".csv":
        return create_csv_report(datacollector, report_path)
    elif ext in [".xls", ".xlsx"]:
        return create_excel_report(datacollector, report_path)
    else:
        logging.info(f"Unsupported report format: {ext}")

def create_excel_report(datacollector: DataCollector, report_path: str):
    df = create_report_dataframe(datacollector)
    
    # Write the DataFrame to an Excel file.
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Report")
    
    logging.info(f"Excel report generated: {report_path}")
    return df

def create_csv_report(datacollector: DataCollector, report_path: str):
    df = create_report_dataframe(datacollector)

    # Write the DataFrame to CSV.
    # Pandas will output the multi-index header in two rows.
    df.to_csv(report_path)
    logging.info(f"CSV report generated: {report_path}")
    return df

def create_json_report(datacollector: DataCollector, report_path: str):
    df = create_report_dataframe(datacollector)

    # Convert MultiIndex DataFrame into desired dict structure
    result = {}
    for col in df.columns:
        end, metric = col
        col_key = f"(end: {end}, '{metric}')"
        result[col_key] = {}
        for idx, value in df[col].items():
            direction, lane = idx
            row_key = f"(direction: {direction}, lane: {lane})"
            result[col_key][row_key] = None if pd.isna(value) else float(value)

    # Save as pretty JSON
    import json
    with open(report_path, "w") as f:
        json.dump(result, f, indent=4)

    logging.info(f"JSON report generated: {report_path}")
    return result