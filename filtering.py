import pandas as pd
from datetime import datetime, timedelta
import re
import os

def filter_recent_reviews(csv_path: str, max_years: int = 2, output_folder: str = "filtered") -> pd.DataFrame:
    """
    Load CSV and filter reviews:
    1. Only include rows with timestamp ≤ `max_years` ago.
    2. Only include rows where caption is not blank.
    3. Save result to an Excel file in `filtered/` folder.

    Args:
        csv_path (str): Input CSV file path.
        max_years (int): Max age of reviews to keep.
        output_folder (str): Folder to save the Excel result.

    Returns:
        pd.DataFrame: Filtered data.
    """
    
    def parse_relative_time(ts: str):
        now = datetime.now()
        if pd.isnull(ts):
            return None
        ts = ts.lower().strip()

        try:
            if "menit" in ts:
                match = re.search(r"(\d+)", ts)
                if match:
                    return now - timedelta(minutes=int(match.group(1)))
            elif "jam" in ts:
                match = re.search(r"(\d+)", ts)
                if match:
                    return now - timedelta(hours=int(match.group(1)))
            elif "hari" in ts:
                match = re.search(r"(\d+)", ts)
                if match:
                    return now - timedelta(days=int(match.group(1)))
            elif "bulan" in ts:
                match = re.search(r"(\d+)", ts)
                if match:
                    return now - timedelta(days=int(match.group(1)) * 30)
            elif "tahun" in ts:
                match = re.search(r"(\d+)", ts)
                if match:
                    return now - timedelta(days=int(match.group(1)) * 365)
                elif "setahun" in ts:
                    return now - timedelta(days=365)
            elif "setahun" in ts:
                return now - timedelta(days=365)
        except Exception:
            return None
        
        return None

    # Load CSV
    df = pd.read_csv(csv_path)

    # Apply timestamp parsing
    df["parsed_timestamp"] = df["timestamp"].apply(parse_relative_time)

    # Filter by max age and non-empty caption
    cutoff_date = datetime.now() - timedelta(days=max_years * 365)
    filtered_df = df[
        (df["parsed_timestamp"] >= cutoff_date) &
        (df["caption"].notnull()) &
        (df["caption"].str.strip() != "")
    ]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Create output Excel filename
    base_filename = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(output_folder, f"{base_filename}_filtered.xlsx")

    # Save to Excel
    filtered_df.to_excel(output_path, index=False)

    print(f"✅ Filtered data saved to: {output_path}")
    return filtered_df

filtered = filter_recent_reviews("source/taman bunga celosia.csv")
