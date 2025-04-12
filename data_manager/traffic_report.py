import pandas as pd
import numpy as np
from collections import defaultdict
from traffic_observer.crossroad_manager import DataCollector

def create_report_dataframe(datacollector: DataCollector):
    # Group data by lane_id and end_id.
    # Each vehicle contributes a tuple of (start_delay, travel_time)
    data_dict = defaultdict(lambda: defaultdict(list))
    for vehicle in datacollector.collected_vehicles:
        if vehicle.end_id is not None:
            data_dict[vehicle.lane_id][vehicle.end_id].append((vehicle.start_delay, vehicle.travel_time))
    
    # Determine sorted unique lanes and ends.
    lanes = sorted(data_dict.keys())
    ends = set()
    for lane in data_dict:
        ends.update(data_dict[lane].keys())
    ends = sorted(ends)
    
    # Prepare a dictionary to hold data for DataFrame construction.
    # For each end, we compute:
    #  - the average start_delay (only including those values >= 10)
    #  - the average travel_time (including all values)
    #  - the vehicle count
    df_data = {}
    for end in ends:
        avg_start_delay = []
        avg_travel_time = []
        vehicle_count = []
        for lane in lanes:
            values = data_dict[lane].get(end)
            if values:
                # Filter out start_delay values < 10.
                valid_delays = [v[0] for v in values if v[0] >= 10]
                if valid_delays:
                    avg_delay = sum(valid_delays) / len(valid_delays)
                else:
                    avg_delay = np.nan
                # For travel_time, use all values.
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
            
        df_data[(end, "start_delay")] = avg_start_delay
        df_data[(end, "travel_time")] = avg_travel_time
        df_data[(end, "vehicle_count")] = vehicle_count

    # Build a MultiIndex for the columns: top-level for end_id, second-level for the metric.
    multi_columns = pd.MultiIndex.from_tuples(list(df_data.keys()), names=["end", ""])
    
    # Construct the DataFrame.
    df = pd.DataFrame(df_data, index=lanes)
    df.index.name = "lane"
    df = df.reindex(columns=multi_columns)

    return df

def create_excel_report(datacollector: DataCollector, report_path: str):
    df = create_report_dataframe(datacollector)
    
    # Write the DataFrame to an Excel file.
    with pd.ExcelWriter(report_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Report")
    
    print(f"Excel report generated: {report_path}")
    return df

def create_csv_report(datacollector: DataCollector, report_path: str):
    df = create_report_dataframe(datacollector)

    # Write the DataFrame to CSV.
    # Pandas will output the multi-index header in two rows.
    df.to_csv(report_path)
    print(f"CSV report generated: {report_path}")
    return df