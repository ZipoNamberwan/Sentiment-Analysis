"""
Script to combine multiple CSV files from the 'places' folder into a single CSV file.
The combined file will be saved in the same 'places' folder.
"""

import pandas as pd
import glob
import os
import re
from datetime import datetime
import math

def calculate_distance_meters(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two coordinates using the Haversine formula.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        float: Distance in meters, or None if any coordinate is missing
    """
    if any(coord is None or pd.isna(coord) for coord in [lat1, lon1, lat2, lon2]):
        return None
    
    try:
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of Earth in meters
        earth_radius_meters = 6371000
        
        # Calculate the distance
        distance = earth_radius_meters * c
        return round(distance, 2)
    
    except (ValueError, TypeError):
        return None

def transform_to_new_url_format(old_url, latitude, longitude):
    """
    Transform old Google Maps URL format to new format using extracted coordinates.
    
    Args:
        old_url (str): Original Google Maps URL
        latitude (float): Extracted latitude
        longitude (float): Extracted longitude
        
    Returns:
        str: New format URL or original URL if transformation fails
    """
    if pd.isna(old_url) or not isinstance(old_url, str):
        return old_url
    
    if latitude is None or longitude is None or pd.isna(latitude) or pd.isna(longitude):
        return old_url
    
    try:
        # Extract the place name from the URL
        # Look for the pattern after /maps/place/ and before the next /
        place_match = re.search(r'/maps/place/([^/]+)/', old_url)
        if not place_match:
            return old_url
        
        place_name = place_match.group(1)
        
        # Extract additional parameters from original URL
        auth_match = re.search(r'authuser=(\d+)', old_url)
        lang_match = re.search(r'hl=([^&]+)', old_url)
        
        # Build new URL format
        new_url = f"https://www.google.com/maps/place/{place_name}/@{latitude},{longitude},17z/data=!4m11!3m10!1s0x2e708743a9420591:0x3addcade69b472a1!5m2!4m1!1i2!8m2!3d{latitude}!4d{longitude}!9m1!1b1!16s%2Fg%2F11h0pks60r"
        
        # Add query parameters
        params = []
        if auth_match:
            params.append(f"authuser={auth_match.group(1)}")
        if lang_match:
            params.append(f"hl={lang_match.group(1)}")
        
        # Add default parameters for new format
        params.extend([
            "entry=ttu",
            "g_ep=EgoyMDI1MDcwOS4wIKXMDSoASAFQAw%3D%3D"
        ])
        
        if params:
            new_url += "?" + "&".join(params)
        
        return new_url
    
    except Exception:
        return old_url

def load_tourism_coordinates(coord_file_path: str = "places/tourism places coordinate.csv"):
    """
    Load tourism places coordinates from the reference file.
    
    Args:
        coord_file_path (str): Path to the tourism places coordinate CSV file
        
    Returns:
        dict: Dictionary mapping place names to coordinates {place_name: (lat, lng)}
    """
    try:
        # Read the coordinate file with semicolon delimiter
        coord_df = pd.read_csv(coord_file_path, delimiter=';')
        
        # Create a dictionary mapping place names to coordinates
        coord_dict = {}
        for _, row in coord_df.iterrows():
            place_name = str(row['places']).strip().lower()
            lat = float(str(row['latitude']).strip())
            lng = float(str(row['longitude']).strip())
            coord_dict[place_name] = (lat, lng)
        
        print(f"📍 Loaded coordinates for {len(coord_dict)} tourism places from reference file")
        return coord_dict
    
    except Exception as e:
        print(f"⚠️ Error loading tourism coordinates: {e}")
        return {}

def get_tourism_coordinates(source_filename: str, coord_dict: dict):
    """
    Get tourism place coordinates from the reference dictionary based on source filename.
    
    Args:
        source_filename (str): Source CSV filename (e.g., 'ayanaz.csv')
        coord_dict (dict): Dictionary of place coordinates
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    if not source_filename or not coord_dict:
        return None, None
    
    # Extract place name from filename (remove .csv extension)
    place_name = str(source_filename).replace('.csv', '').strip().lower()
    
    # Direct match
    if place_name in coord_dict:
        return coord_dict[place_name]
    
    # Fuzzy matching - check if any reference place name is contained in the filename
    for ref_place, coords in coord_dict.items():
        if ref_place in place_name or place_name in ref_place:
            return coords
    
    return None, None

def extract_coordinates_from_link(link):
    """
    Extract latitude and longitude coordinates from Google Maps link.
    
    Args:
        link (str): Google Maps URL containing coordinates
        
    Returns:
        tuple: (latitude, longitude) or (None, None) if not found
    """
    if pd.isna(link) or not isinstance(link, str):
        return None, None
    
    # Pattern to match coordinates in Google Maps URLs
    # Looks for patterns like: 3d-7.212555!4d110.3444924
    coord_pattern = r'3d(-?\d+\.?\d*)!4d(-?\d+\.?\d*)'
    
    match = re.search(coord_pattern, link)
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        return latitude, longitude
    
    # Alternative pattern for different URL formats
    # Looks for patterns like: @-7.212555,110.3444924
    alt_pattern = r'@(-?\d+\.?\d*),(-?\d+\.?\d*)'
    
    match = re.search(alt_pattern, link)
    if match:
        latitude = float(match.group(1))
        longitude = float(match.group(2))
        return latitude, longitude
    
    return None, None

def combine_csv_files(folder_path: str = "places", output_filename: str = "combined_places.csv"):
    """
    Combine specific columns from all CSV files in the specified folder into a single CSV file.
    Only includes: place_id, name, reviews, rating, main_category, categories, address, link, query, latitude, longitude, tourism_place_lat, tourism_place_long, distance_meters, new_url
    
    Args:
        folder_path (str): Path to the folder containing CSV files (default: "places")
        output_filename (str): Name of the output combined file (default: "combined_places.csv")
    
    Returns:
        str: Path to the combined CSV file
    """
    
    # Define the specific columns to include
    required_columns = ['place_id', 'name', 'reviews', 'rating', 'main_category', 'categories', 'address', 'link', 'query', 'latitude', 'longitude', 'tourism_place_lat', 'tourism_place_long', 'distance_meters', 'new_url']
    
    # Load tourism coordinates reference data
    tourism_coords = load_tourism_coordinates(os.path.join(folder_path, "result", "tourism places coordinate.csv"))
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return None
    
    # Find all CSV files in the folder
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Create result folder if it doesn't exist
    result_folder = os.path.join(folder_path, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"📁 Created result folder: {result_folder}")
    
    # Set output path to result folder
    output_path = os.path.join(result_folder, output_filename)
    
    # Remove any files that contain "combined" or are in result folder to avoid recursive processing
    csv_files = [f for f in csv_files if not (
        "combined" in os.path.basename(f).lower() or 
        f == output_path or
        "result" in f.lower() or
        "tourism places coordinate" in os.path.basename(f).lower()
    )]
    
    if not csv_files:
        print(f"❌ No CSV files found in the '{folder_path}' folder.")
        return None
    
    print(f"📁 Found {len(csv_files)} CSV file(s) in the '{folder_path}' folder:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    print()
    print(f"🎯 Extracting specific columns: {', '.join(required_columns)}")
    print()
    
    # Initialize list to store dataframes
    dataframes = []
    file_info = []
    missing_columns_report = []
    
    # Read each CSV file and combine them
    for csv_file in csv_files:
        try:
            print(f"📖 Reading: {os.path.basename(csv_file)}")
            df = pd.read_csv(csv_file)
            
            # Check which required columns are available
            available_columns = [col for col in required_columns if col in df.columns]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ⚠️ Missing columns in {os.path.basename(csv_file)}: {', '.join(missing_columns)}")
                missing_columns_report.append({
                    'file': os.path.basename(csv_file),
                    'missing': missing_columns
                })
            
            if not available_columns:
                print(f"  ❌ No required columns found in {os.path.basename(csv_file)}")
                continue
            
            # Select only the available required columns
            df_selected = df[available_columns].copy()
            
            # Add missing columns with NaN values
            for missing_col in missing_columns:
                df_selected[missing_col] = None
            
            # Reorder columns to match the required order
            df_selected = df_selected.reindex(columns=required_columns)
            
            # Extract coordinates from link column if it exists
            if 'link' in df_selected.columns:
                print(f"  🗺️ Extracting coordinates from links...")
                coordinates = df_selected['link'].apply(extract_coordinates_from_link)
                df_selected['latitude'] = coordinates.apply(lambda x: x[0])
                df_selected['longitude'] = coordinates.apply(lambda x: x[1])
                
                # Count successful coordinate extractions
                valid_coords = df_selected[['latitude', 'longitude']].dropna()
                print(f"  📍 Extracted coordinates for {len(valid_coords)}/{len(df_selected)} rows")
            else:
                # Add empty coordinate columns if link column is missing
                df_selected['latitude'] = None
                df_selected['longitude'] = None
                print(f"  ⚠️ No link column found - coordinates set to None")
            
            # Add tourism place coordinates from reference file
            print(f"  🏛️ Adding tourism place coordinates...")
            source_filename = os.path.basename(csv_file)
            tourism_lat, tourism_lng = get_tourism_coordinates(source_filename, tourism_coords)
            
            # Apply the same coordinates to all rows in this file
            df_selected['tourism_place_lat'] = tourism_lat
            df_selected['tourism_place_long'] = tourism_lng
            
            # Calculate distance between coordinates
            print(f"  📏 Calculating distances between coordinates...")
            df_selected['distance_meters'] = df_selected.apply(
                lambda row: calculate_distance_meters(
                    row['latitude'], row['longitude'],
                    row['tourism_place_lat'], row['tourism_place_long']
                ), axis=1
            )
            
            # Count successful distance calculations
            valid_distances = df_selected['distance_meters'].dropna()
            if len(valid_distances) > 0:
                avg_distance = valid_distances.mean()
                print(f"  📍 Calculated distances for {len(valid_distances)}/{len(df_selected)} rows (avg: {avg_distance:.0f}m)")
            else:
                print(f"  ⚠️ No valid distances calculated")
            
            # Transform URLs to new format
            print(f"  🔗 Transforming URLs to new format...")
            df_selected['new_url'] = df_selected.apply(
                lambda row: transform_to_new_url_format(
                    row['link'], row['latitude'], row['longitude']
                ), axis=1
            )
            
            # Count successful URL transformations
            if 'link' in df_selected.columns:
                transformed_urls = df_selected[df_selected['new_url'] != df_selected['link']]
                print(f"  🔄 Transformed {len(transformed_urls)}/{len(df_selected)} URLs to new format")
            else:
                print(f"  ⚠️ No link column found - new_url set to None")
            
            # Count successful tourism coordinate matches
            if tourism_lat is not None and tourism_lng is not None:
                print(f"  🎯 Applied tourism coordinates ({tourism_lat}, {tourism_lng}) to all {len(df_selected)} rows")
            else:
                print(f"  ⚠️ No tourism coordinates found for {source_filename}")
            
            # Add source file column to track origin
            df_selected['source_file'] = os.path.basename(csv_file)
            
            dataframes.append(df_selected)
            file_info.append({
                'filename': os.path.basename(csv_file),
                'rows': len(df_selected),
                'available_columns': len(available_columns),
                'missing_columns': len(missing_columns)
            })
            
            print(f"  ✅ Added {len(df_selected)} rows with {len(available_columns)}/{len(required_columns)} columns")
            
        except Exception as e:
            print(f"⚠️ Error reading {os.path.basename(csv_file)}: {e}")
            continue
    
    if not dataframes:
        print("❌ No valid CSV files could be read.")
        return None
    
    # Combine all dataframes
    print("\n🔄 Combining CSV files...")
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    # Save the combined dataframe
    combined_df.to_csv(output_path, index=False)
    
    # Generate summary report
    print("\n📊 COMBINATION SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully combined {len(dataframes)} CSV files")
    print(f"📄 Total rows in combined file: {len(combined_df)}")
    print(f"📄 Columns in combined file: {len(combined_df.columns)} ({', '.join(combined_df.columns)})")
    print(f"💾 Combined file saved as: {output_path}")
    
    print("\n📋 Individual file details:")
    total_rows = 0
    for info in file_info:
        print(f"  📄 {info['filename']}: {info['rows']} rows, {info['available_columns']}/{len(required_columns)} required columns")
        total_rows += info['rows']
    
    if missing_columns_report:
        print("\n⚠️ Missing columns report:")
        for report in missing_columns_report:
            print(f"  📄 {report['file']}: Missing {', '.join(report['missing'])}")
    
    print(f"\n🔢 Total rows from all files: {total_rows}")
    print(f"🔢 Rows in combined file: {len(combined_df)}")
    
    if total_rows == len(combined_df):
        print("✅ All rows successfully combined!")
    else:
        print("⚠️ Row count mismatch - some data may have been lost during combination")
    
    return output_path

def combine_with_metadata(folder_path: str = "places", output_filename: str = "combined_places_with_metadata.csv"):
    """
    Combine specific columns from CSV files and add additional metadata columns.
    Only includes: place_id, name, reviews, rating, main_category, categories, address, link, query, latitude, longitude, tourism_place_lat, tourism_place_long, distance_meters, new_url
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        output_filename (str): Name of the output combined file
    
    Returns:
        str: Path to the combined CSV file with metadata
    """
    
    # Define the specific columns to include
    required_columns = ['place_id', 'name', 'reviews', 'rating', 'main_category', 'categories', 'address', 'link', 'query', 'latitude', 'longitude', 'tourism_place_lat', 'tourism_place_long', 'distance_meters', 'new_url']
    
    # Load tourism coordinates reference data
    tourism_coords = load_tourism_coordinates(os.path.join(folder_path, "result", "tourism places coordinate.csv"))
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Error: Folder '{folder_path}' not found.")
        return None
    
    # Find all CSV files
    csv_pattern = os.path.join(folder_path, "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    # Create result folder if it doesn't exist
    result_folder = os.path.join(folder_path, "result")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
        print(f"📁 Created result folder: {result_folder}")
    
    # Set output path to result folder
    output_path = os.path.join(result_folder, output_filename)
    
    # Remove any files that contain "combined" or are in result folder to avoid recursive processing
    csv_files = [f for f in csv_files if not (
        "combined" in os.path.basename(f).lower() or 
        f == output_path or
        "result" in f.lower() or
        "tourism places coordinate" in os.path.basename(f).lower()
    )]
    
    if not csv_files:
        print(f"❌ No CSV files found in the '{folder_path}' folder.")
        return None
    
    print(f"📁 Combining {len(csv_files)} CSV files with metadata...")
    print(f"🎯 Extracting specific columns: {', '.join(required_columns)}")
    print()
    
    dataframes = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check which required columns are available
            available_columns = [col for col in required_columns if col in df.columns]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ⚠️ Missing columns in {os.path.basename(csv_file)}: {', '.join(missing_columns)}")
            
            if not available_columns:
                print(f"  ❌ No required columns found in {os.path.basename(csv_file)}")
                continue
            
            # Select only the available required columns
            df_selected = df[available_columns].copy()
            
            # Add missing columns with NaN values
            for missing_col in missing_columns:
                df_selected[missing_col] = None
            
            # Reorder columns to match the required order
            df_selected = df_selected.reindex(columns=required_columns)
            
            # Extract coordinates from link column if it exists
            if 'link' in df_selected.columns:
                coordinates = df_selected['link'].apply(extract_coordinates_from_link)
                df_selected['latitude'] = coordinates.apply(lambda x: x[0])
                df_selected['longitude'] = coordinates.apply(lambda x: x[1])
            else:
                df_selected['latitude'] = None
                df_selected['longitude'] = None
            
            # Add tourism place coordinates from reference file
            source_filename = os.path.basename(csv_file)
            tourism_lat, tourism_lng = get_tourism_coordinates(source_filename, tourism_coords)
            
            # Apply the same coordinates to all rows in this file
            df_selected['tourism_place_lat'] = tourism_lat
            df_selected['tourism_place_long'] = tourism_lng
            
            # Calculate distance between coordinates
            df_selected['distance_meters'] = df_selected.apply(
                lambda row: calculate_distance_meters(
                    row['latitude'], row['longitude'],
                    row['tourism_place_lat'], row['tourism_place_long']
                ), axis=1
            )
            
            # Transform URLs to new format
            df_selected['new_url'] = df_selected.apply(
                lambda row: transform_to_new_url_format(
                    row['link'], row['latitude'], row['longitude']
                ), axis=1
            )
            
            # Add metadata columns
            df_selected['source_file'] = os.path.basename(csv_file)
            df_selected['file_size_kb'] = round(os.path.getsize(csv_file) / 1024, 2)
            df_selected['combined_date'] = current_time
            df_selected['original_row_number'] = range(1, len(df_selected) + 1)
            df_selected['columns_available'] = len(available_columns)
            df_selected['columns_missing'] = len(missing_columns)
            
            dataframes.append(df_selected)
            print(f"  ✅ Added: {os.path.basename(csv_file)} ({len(df_selected)} rows, {len(available_columns)}/{len(required_columns)} columns)")
            
        except Exception as e:
            print(f"  ❌ Error: {os.path.basename(csv_file)} - {e}")
            continue
    
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
        combined_df.to_csv(output_path, index=False)
        
        print(f"\n✅ Enhanced combination completed!")
        print(f"📄 Combined file: {output_path}")
        print(f"📊 Total rows: {len(combined_df)}")
        print(f"📊 Total columns: {len(combined_df.columns)}")
        print(f"📊 Required columns: {', '.join(required_columns)}")
        print(f"📊 Metadata columns: source_file, file_size_kb, combined_date, original_row_number, columns_available, columns_missing")
        
        return output_path
    else:
        print("❌ No files could be combined.")
        return None

def main():
    """
    Main function to combine CSV files in the places folder.
    """
    print("🚀 CSV File Combiner for Places Folder")
    print("=" * 50)
    
    # Standard combination
    print("📋 Option 1: Standard combination")
    result1 = combine_csv_files("places", "combined_places.csv")
    
    if result1:
        print(f"\n🎉 Standard combination successful: {result1}")
        
        # Ask if user wants enhanced version with metadata
        print("\n" + "=" * 50)
        print("📋 Option 2: Enhanced combination with metadata")
        result2 = combine_with_metadata("places", "combined_places_with_metadata.csv")
        
        if result2:
            print(f"🎉 Enhanced combination successful: {result2}")
    
    print("\n🏁 CSV combination process completed!")

if __name__ == "__main__":
    main()