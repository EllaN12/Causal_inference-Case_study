#%%
import os
from bs4 import BeautifulSoup as soup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

#for causal_grapgh
#from causalgraphicalmodels import CausalGraphicalModel
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import Dict, List, Set, Tuple
import logging, sys
import pandas_flavor as pf
from geopy.distance import geodesic
import seaborn as sns   
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import networkx as nx
import ydata_profiling as yd
from pathlib import Path
import matplotlib.pyplot as plt 
from matplotlib.patches import Patch
import sys

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _MODULE_DIR.parent

# Make sure both pipeline locations are importable from this folder.
for candidate in [_PROJECT_DIR, _PROJECT_DIR / "venv", _PROJECT_DIR / "01_Data_Analysis"]:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from data_pipeline import run_pipeline
from causal_analysis_module.analysis import data_preprocessing


try:
    # Preferred source: unified ETL pipeline output.
    df = run_pipeline(save_output=True, run_eda_report=False)
except Exception:
    # Fallback for legacy behavior if pipeline run fails.
    df = data_preprocessing()
eda_report = yd.ProfileReport(df, title="Profiling_Report", explorative=True)
eda_report.to_file(os.path.abspath("Reports/eda_report.html"))


def webscrape_report():
    """
    Web scrapes a Ydata EDA HTML report to extract information about data quality alerts,
    specifically focusing on high correlations.
    
    Tries to use WebDriver first, but falls back to direct file reading if WebDriver fails.

    Returns:
        A dictionary where keys are variable names with high correlation alerts,
        and values are the corresponding "other fields" information.
    """

    file_path = os.path.abspath("Reports/eda_report.html")
    correlation_dict = {}
    html = None
    driver = None

    # Try to use WebDriver first (for JavaScript-rendered content)
    try:
        file_url = "file://" + file_path
        print(f"Attempting to use WebDriver...")
        
        executable_path = ChromeDriverManager().install()
        service = Service(executable_path)
        options = Options()

        # Options to try to avoid detection 
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')

        driver = webdriver.Chrome(service=service, options=options)
        driver.get(file_url)
        html = driver.page_source
        print("✓ Successfully loaded page with WebDriver")
        
    except Exception as e:
        print(f"⚠ WebDriver failed: {e}")
        print("Falling back to direct file reading...")
        driver = None
    
    # Fallback to direct file reading if WebDriver failed or wasn't used
    if html is None:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                html = file.read()
            print("✓ Successfully read file directly")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return correlation_dict
        except Exception as e:
            print(f"Error reading file: {e}")
            return correlation_dict
    
    # Clean up WebDriver if it was used
    if driver is not None:
        try:
            driver.quit()
        except:
            pass
    
    # Parse the HTML
    try:
        soupy = soup(html, 'html.parser')
        alerts_table = soupy.find('div', id='tab-pane-overview-alerts')
        
        if alerts_table:
            table = alerts_table.find('table', class_='table-striped')
            
            if table:
                for row in table.find_all('tr'):
                    cells = row.find_all('td')
                    if cells and len(cells) > 0:
                        # Extract variable name from first <code> tag
                        first_code = cells[0].find('code')
                        if first_code:
                            variable_name = first_code.text.strip()
                            
                            # Extract tooltip data
                            tooltip_span = cells[0].find('span', attrs={'data-bs-toggle': 'tooltip'})
                            if tooltip_span and 'data-bs-title' in tooltip_span.attrs:
                                tooltip_data = tooltip_span['data-bs-title']
                                
                                # Check if it's a correlation alert
                                if 'highly overall correlated' in cells[0].text:
                                    correlation_dict[variable_name] = tooltip_data
                                    
                                    print(f"Variable: {variable_name}")
                                    print(f"Correlated Fields: {tooltip_data}")
                                    print("-" * 50)
            else:
                print("Table not found within alerts div.")
        else:
            print("Alerts div not found.")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary of high correlations after scraping
    print("\nHigh Correlation Summary:")
    for variable_name, fields in correlation_dict.items():
        print(f"Variable: {variable_name}, Other Fields: {fields}")

    return correlation_dict

        
        

d = webscrape_report()

print(d)

#%%


#