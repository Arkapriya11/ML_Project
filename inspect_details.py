import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

files = {
    'AIIMS_Hourly': r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\DCR AIIMS\2023\jan-23\DCR AIIMS_RAIPUR_ Hourly - JAN-2023.xls',
    'AIIMS_Quat':   r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\DCR AIIMS\2023\jan-23\DCR AIIMS_RAIPUR _ QUAT - JAN -2023.xls',
    'Siltara_Hourly': r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\SILTARA DCR\2023\JAN 2023\DCR SILTARA (RAIPUR)_ Hourly JAN..-2023.xlsb',
    'Siltara_Quat':   r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\SILTARA DCR\2023\JAN 2023\DCR SILTARA _ QUAT -JAN.-2023.xls',
    'IGKV_Hourly':    r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\IGKV DCR\Year 2023\JANUARY 2023\JANUARY 2023 - Hourly - IGKV.xlsb',
    'IGKV_Quat':      r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\IGKV DCR\Year 2023\JANUARY 2023\JANUARY 2023 - Quat. Hourly - IGKV.xlsb',
    'Bhatagaon_Hourly': r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\Bhatagaon DCR\DCR bhatagaon\DCR 2023\Jan.2023\DCR BHATAGAON (RAIPUR)_ Hourly  Jan. 2023.xlsx',
    'Bhatagaon_Quat':   r'c:\Users\annam\OneDrive\Desktop\ML_Project\Datasets\Bhatagaon DCR\DCR bhatagaon\DCR 2023\Jan.2023\DCR Bhatagaon _ QUAT - Jan. 2023.xlsx',
}

def inspect_file(name, path):
    print(f"\n{'='*20} {name} {'='*20}")
    try:
        if path.endswith('.xlsb'): engine = 'pyxlsb'
        elif path.endswith('.xls'): engine = 'xlrd'
        else: engine = 'openpyxl'
        
        df = pd.read_excel(path, header=None, engine=engine)
        print(f"Path: {path}")
        print(f"Shape: {df.shape}")
        
        # Look for the row that contains 'DateTime' or pollutant names to identify header
        potential_headers = []
        for i in range(min(30, len(df))):
            row_values = [str(x).strip().lower() for x in df.iloc[i] if pd.notnull(x)]
            if any('date' in x or 'time' in x or 'pm2.5' in x or 'no2' in x or 'so2' in x for x in row_values):
                potential_headers.append(i)
                print(f"Potential Header @ Row {i}: {list(df.iloc[i])}")
        
        if not potential_headers:
            print("No obvious header found in first 30 rows.")
        
        print("Data sample (Rows 15-25):")
        print(df.iloc[15:25, :10].to_string(header=False))
        
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

for name, path in files.items():
    inspect_file(name, path)
