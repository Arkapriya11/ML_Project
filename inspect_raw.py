import pandas as pd
import warnings
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

for name, path in files.items():
    print(f'\n{"="*60}')
    print(f'FILE: {name}')
    print(f'Path: {path}')
    try:
        if path.endswith('.xlsb'):
            df = pd.read_excel(path, header=None, engine='pyxlsb')
        elif path.endswith('.xls'):
            df = pd.read_excel(path, header=None, engine='xlrd')
        else:
            df = pd.read_excel(path, header=None, engine='openpyxl')
        print(f'Shape: {df.shape}')
        print('First 20 rows:')
        for i in range(min(20, len(df))):
            row = list(df.iloc[i])
            print(f'  Row {i:2d}: {row}')
    except Exception as e:
        print(f'ERROR: {e}')
