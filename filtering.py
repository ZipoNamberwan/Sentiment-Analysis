import pandas as pd
from datetime import datetime, timedelta
import re
import os

class DataFilter:
    """
    A class to filter CSV data based on timestamp and caption criteria.
    Purpose: Generate filtered Excel files from source data.
    """
    
    def __init__(self, max_years: int = 2, output_folder: str = "filtered"):
        """
        Initialize the DataFilter.
        
        Args:
            max_years (int): Maximum age of reviews to keep (default: 2 years)
            output_folder (str): Folder to save filtered results (default: "filtered")
        """
        self.max_years = max_years
        self.output_folder = output_folder
        
    def _parse_relative_time(self, ts: str):
        """Parse Indonesian relative time strings to datetime objects."""
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
        
    def filter_recent_reviews(self, csv_path: str) -> pd.DataFrame:
        """
        Filter CSV data based on timestamp and caption criteria.
        
        Args:
            csv_path (str): Input CSV file path
            
        Returns:
            pd.DataFrame: Filtered data
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Apply timestamp parsing
        df["parsed_timestamp"] = df["timestamp"].apply(self._parse_relative_time)

        # Filter by max age and non-empty caption
        cutoff_date = datetime.now() - timedelta(days=self.max_years * 365)
        filtered_df = df[
            (df["parsed_timestamp"] >= cutoff_date) &
            (df["caption"].notnull()) &
            (df["caption"].str.strip() != "")
        ]

        return filtered_df
    
    def generate_filtered_csv(self, csv_path: str) -> str:
        """
        Generate a filtered Excel file from source data.
        
        Args:
            csv_path (str): Input CSV file path
            
        Returns:
            str: Path to the generated filtered file
        """
        # Get filtered data
        filtered_df = self.filter_recent_reviews(csv_path)
        
        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

        # Create output Excel filename
        base_filename = os.path.splitext(os.path.basename(csv_path))[0]
        output_path = os.path.join(self.output_folder, f"{base_filename}_filtered.xlsx")

        # Save to Excel
        filtered_df.to_excel(output_path, index=False)

        print(f"✅ Filtered data saved to: {output_path}")
        return output_path

# Example usage
if __name__ == "__main__":
    filter_processor = DataFilter(max_years=2, output_folder="filtered")
    output_file = filter_processor.generate_filtered_csv("source/taman bunga celosia.csv")
