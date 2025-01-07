import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict
from serpapi import GoogleSearch
import json
import csv
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import io

def data_collection_pipeline():
    collect_oecd_data(directory="../04 Data/OECD/Raw")
    collect_stocks(yf_stocks_directory = "../04 Data/Stock Movements Yahoo Finance/Raw", 
                   time_filename_extension = "alltime", stocknames=stock_names)
def data_integration_pipeline():
    # OECD
    process_all_oecd_data()
    integrate_and_optimize_oecd_data()
    # Order Intake
    hg_oi_series_monthly_df = preprocess_hg_df(input_filepath="../04 Data/HOMAG/Raw/OrderIntake_regional_monthly.CSV", output_filepath="../04 Data/HOMAG/OrderIntake_series_monthly.csv", homag_bu_dict=homag_bu_dict)
    integrate_stocks_data(stock_names=stock_names)


# OECD
oecd_categorical_codes = {
    "MEASURE": {
        "TOT": "Total",
        "AGRWTH": "Annual Growth",
        "IDX2015": "Index base year 2015",
        "PC_GDP": "Percentage of GDP",
        "DEU": "Domestic extraction used",
        "IMP": "Imports",
        "EXP": "Exports",
        "PTB": "Physical Trade Balance",
        "DMI": "Direct Material Input",
        "DMC": "Domestic Material Consumption",
        "MF": "Material footprint"
    },
    "ACTIVITY": {
        "BTE": "Industry (except construction)",
        "C": "Manufacturing",
        "F": "Construction"
    },
    "ADJUSTMENT": {
        "Y": "Calendar and seasonally adjusted",
        "N": "Neither seasonally adjusted nor calendar adjusted"
    },
    "OBS_STATUS": {
        "A": "Normal value",
        "E": "Estimated value",
        "F": "Forecast",
        "I": "Interpolated",
        "M": "Missing value",
        "P": "Provisional value",
        "S": "Strike", # missing because of strike
        "T": "Break in series",
        "U": "Low reliability",
        "W": "Working day adjusted"
    },
    "FREQ": {
        "A": "Annual",
        "Q": "Quarterly",
        "M": "Monthly",
        "W": "Weekly"
    },
    "UNIT_MEASURE": {
        "USD": "US Dollars",
        "IX": "Index",
        "EUR": "Euros",
        "PERS": "Per Person",
        "THND": "Thousands",
        "T": "Tonnes",
        "USD_T": "US dollars per tonne",
        "T_PS": "Tonnes per person",
        "PT_CONS_MAT_D": "Percentage of domestic material consumption"
    },
    "GROUP": {
        "TOT": "Total",
        "BIO": "Biomass",
        "FOOD": "Biomass for food and feed",
        "WOOD": "Wood",
        "FUEL": "Fossil energy materials/carriers",
        "IND": "Non-metallic minerals",
        "CONST": "Construction minerals",
        "MIN": "Other non-metallic minerals",
        "MET": "Metals"
    },
    "REF_AREA": {
        "EU27_2020": "European Union (27 countries from 01/02/2020)", 
        "EA19": "Euro Area (19 countries)", 
        "OECD": "OECD", 
        "OECDA": "OECD America", 
        "OECDSO": "OECD Asia Oceania", 
        "OECDE": "OECD Europe", 
        "WXOECD": "Non-OECD economies",
        "AFG": "Afghanistan",
        "ALB": "Albania",
        "DZA": "Algeria",
        "AND": "Andorra",
        "AGO": "Angola",
        "ATG": "Antigua and Barbuda",
        "ARG": "Argentina",
        "ARM": "Armenia",
        "AUS": "Australia",
        "AUT": "Austria",
        "AZE": "Azerbaijan",
        "BHS": "Bahamas",
        "BHR": "Bahrain",
        "BGD": "Bangladesh",
        "BRB": "Barbados",
        "BLR": "Belarus",
        "BEL": "Belgium",
        "BLZ": "Belize",
        "BEN": "Benin",
        "BTN": "Bhutan",
        "BOL": "Bolivia",
        "BIH": "Bosnia and Herzegovina",
        "BWA": "Botswana",
        "BRA": "Brazil",
        "BRN": "Brunei",
        "BGR": "Bulgaria",
        "BFA": "Burkina Faso",
        "BDI": "Burundi",
        "CPV": "Cabo Verde",
        "KHM": "Cambodia",
        "CMR": "Cameroon",
        "CAN": "Canada",
        "CAF": "Central African Republic",
        "TCD": "Chad",
        "CHL": "Chile",
        "CHN": "China",
        "COL": "Colombia",
        "COM": "Comoros",
        "COG": "Congo",
        "CRI": "Costa Rica",
        "HRV": "Croatia",
        "CUB": "Cuba",
        "CYP": "Cyprus",
        "CZE": "Czech Republic",
        "DNK": "Denmark",
        "DJI": "Djibouti",
        "DMA": "Dominica",
        "DOM": "Dominican Republic",
        "ECU": "Ecuador",
        "EGY": "Egypt",
        "SLV": "El Salvador",
        "GNQ": "Equatorial Guinea",
        "ERI": "Eritrea",
        "EST": "Estonia",
        "SWZ": "Eswatini",
        "ETH": "Ethiopia",
        "FJI": "Fiji",
        "FIN": "Finland",
        "FRA": "France",
        "GAB": "Gabon",
        "GMB": "Gambia",
        "GEO": "Georgia",
        "DEU": "Germany",
        "GHA": "Ghana",
        "GRC": "Greece",
        "GRD": "Grenada",
        "GTM": "Guatemala",
        "GIN": "Guinea",
        "GNB": "Guinea-Bissau",
        "GUY": "Guyana",
        "HTI": "Haiti",
        "HND": "Honduras",
        "HUN": "Hungary",
        "ISL": "Iceland",
        "IND": "India",
        "IDN": "Indonesia",
        "IRN": "Iran",
        "IRQ": "Iraq",
        "IRL": "Ireland",
        "ISR": "Israel",
        "ITA": "Italy",
        "JAM": "Jamaica",
        "JPN": "Japan",
        "JOR": "Jordan",
        "KAZ": "Kazakhstan",
        "KEN": "Kenya",
        "KIR": "Kiribati",
        "KWT": "Kuwait",
        "KGZ": "Kyrgyzstan",
        "LAO": "Laos",
        "LVA": "Latvia",
        "LBN": "Lebanon",
        "LSO": "Lesotho",
        "LBR": "Liberia",
        "LBY": "Libya",
        "LIE": "Liechtenstein",
        "LTU": "Lithuania",
        "LUX": "Luxembourg",
        "MDG": "Madagascar",
        "MWI": "Malawi",
        "MYS": "Malaysia",
        "MDV": "Maldives",
        "MLI": "Mali",
        "MLT": "Malta",
        "MHL": "Marshall Islands",
        "MRT": "Mauritania",
        "MUS": "Mauritius",
        "MEX": "Mexico",
        "FSM": "Micronesia",
        "MDA": "Moldova",
        "MCO": "Monaco",
        "MNG": "Mongolia",
        "MNE": "Montenegro",
        "MAR": "Morocco",
        "MOZ": "Mozambique",
        "MMR": "Myanmar",
        "NAM": "Namibia",
        "NRU": "Nauru",
        "NPL": "Nepal",
        "NLD": "Netherlands",
        "NZL": "New Zealand",
        "NIC": "Nicaragua",
        "NER": "Niger",
        "NGA": "Nigeria",
        "PRK": "North Korea",
        "MKD": "North Macedonia",
        "NOR": "Norway",
        "OMN": "Oman",
        "PAK": "Pakistan",
        "PLW": "Palau",
        "PAN": "Panama",
        "PNG": "Papua New Guinea",
        "PRY": "Paraguay",
        "PER": "Peru",
        "PHL": "Philippines",
        "POL": "Poland",
        "PRT": "Portugal",
        "QAT": "Qatar",
        "ROU": "Romania",
        "RUS": "Russia",
        "RWA": "Rwanda",
        "KNA": "Saint Kitts and Nevis",
        "LCA": "Saint Lucia",
        "VCT": "Saint Vincent and the Grenadines",
        "WSM": "Samoa",
        "SMR": "San Marino",
        "STP": "Sao Tome and Principe",
        "SAU": "Saudi Arabia",
        "SEN": "Senegal",
        "SRB": "Serbia",
        "SYC": "Seychelles",
        "SLE": "Sierra Leone",
        "SGP": "Singapore",
        "SVK": "Slovakia",
        "SVN": "Slovenia",
        "SLB": "Solomon Islands",
        "SOM": "Somalia",
        "ZAF": "South Africa",
        "KOR": "South Korea",
        "SSD": "South Sudan",
        "ESP": "Spain",
        "LKA": "Sri Lanka",
        "SDN": "Sudan",
        "SUR": "Suriname",
        "SWE": "Sweden",
        "CHE": "Switzerland",
        "SYR": "Syria",
        "TWN": "Taiwan",
        "TJK": "Tajikistan",
        "TZA": "Tanzania",
        "THA": "Thailand",
        "TLS": "Timor-Leste",
        "TGO": "Togo",
        "TON": "Tonga",
        "TTO": "Trinidad and Tobago",
        "TUN": "Tunisia",
        "TUR": "Turkey",
        "TKM": "Turkmenistan",
        "TUV": "Tuvalu",
        "UGA": "Uganda",
        "UKR": "Ukraine",
        "ARE": "United Arab Emirates",
        "GBR": "United Kingdom",
        "USA": "United States",
        "URY": "Uruguay",
        "UZB": "Uzbekistan",
        "VUT": "Vanuatu",
        "VEN": "Venezuela",
        "VNM": "Vietnam",
        "YEM": "Yemen",
        "ZMB": "Zambia",
        "ZWE": "Zimbabwe"
    },
}
oecd_relevant_regions = [
    "USA", 
    "DEU", 
    "CHN", 
    "CHE",
    "AUT",
    #"BRA", 
    #"IND", 
    #"IDN", 
    # "CND", 
    # "ESP", 
    # "PRT", 
    # "FRA", 
    # "GBR", 
    # "ITA", 
    # "SWS", 
    # "AUT", 
    # "RUS", 
    # "UKR", 
    "POL", 
    # "AUS", 
    "DNK", 
    # "FIN", 
    # "GRC", 
    # "JPN", 
    # "NZL", 
    # "NOR", 
    "SWE", 
    "EU27_2020", 
    "EA19", 
    "OECD", 
    "OECDA", 
    "OECDSO", 
    "OECDE", 
    "WXOECD"
    ]
oecd_sources = {
    "MEI_FIN": {
        "Name":"Monthly Monetary and Financial Statistics",
        "SDMX Query": "OECD.SDD.STES,DSD_STES@DF_FINMARK",
        "Measurements":{
            "IRLT": {
                "Name": "Long-Term Interest Rates",
                "Filters": {
                    "MEASURE": "IRLT",
                    "FREQ": "M"
                }
            },
            "IRST": {
                "Name": "Short-Term Interest Rates",
                "Filters":{
                    "MEASURE": "IR3TIB", 
                    "FREQ": "M"
                }
            },
            "IRII": {
                "Name": "Immediate Interest Rates / Interbank Interest Rates",
                "Filters": {
                    "MEASURE": "IRSTCI", 
                    "FREQ": "M"
                }
            },
            "IRN": {
                "Name": "Nominal Interest Rates",
                "Filters": {
                    "MEASURE": "CC", 
                    "FREQ": "M"
                }
            },
            "IRRE": {
                "Name": "Real Effective Exchange Rates - CPI based",
                "Filters": {
                    "MEASURE": "CCRE", 
                    "FREQ": "M"
                }
            }
        }
    },
    "QNA": {
        "Name": "Quarterly National Accounts",
        "SDMX Query": "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD",
        "Measurements": {
            "GDP": {
                "Name": "Gross Domestic Product",
                "Filters": {
                    "TRANSACTION": "B1GQ",
                    "TRANSFORMATION": "G1", 
                    "FREQ":"Q", 
                    "OBS_STATUS": "A"
                }
            }
        }
    },
    "MEI_REAL": {
        "Name": "Production and Sales",
        "SDMX Query": "OECD.SDD.STES,DSD_STES@DF_INDSERV",
        "Measurements": {
            "PRI": {
                "Name": "Production in Industry",
                "Filters": {
                    "ACTIVITY": "BTE",
                    "FREQ":"M", 
                    "ADJUSTMENT": "Y", 
                    "OBS_STATUS": "A"
                },
            },
            "PRM": {
                "Name": "Production in Manufacturing",
                "Filters": {
                    "ACTIVITY": "C",
                    "FREQ":"M", 
                    "ADJUSTMENT": "Y", 
                    "OBS_STATUS": "A"
                }
            },
            "PRC": {
                "Name": "Production in Construction",
                "Filters": {
                    "ACTIVITY": "F",
                    "FREQ":"M", 
                    "ADJUSTMENT": "Y", 
                    "OBS_STATUS": "A"
                }
            }
        }
    },
    "MEI_CLI":{
        "Name": "Main Economic Indicators - Composite Leading Indicators",
        "SDMX Query": "OECD.SDD.STES,DSD_STES@DF_CLI",
        "Measurements":{
            "CLI": {
                "Name": "Composite Leading Indicator",
                "Filters": {
                    "MEASURE": "LI",
                    "ADJUSTMENT": "AA", 
                    "METHODOLOGY":"H", 
                    "FREQ": "M"
                }
            },
            "CBC": {
                "Name": "Composite Business Confidence",
                "Filters": {
                    "MEASURE": "BCICP",
                    "ADJUSTMENT": "AA", 
                    "METHODOLOGY": "H", 
                    "FREQ": "M"
                }
            },# Composite Consumer Confidence (CCICP) "ADJUSTMENT":"AA", "METHODOLOGY":"H", "FREQ":"M"
            "CCP":{
                "Name": "Composite Consumer Confidence",
                "Filters": {
                    "MEASURE": "CCICP",
                    "ADJUSTMENT":"AA", 
                    "METHODOLOGY":"H", 
                    "FREQ":"M"
                }
            }
        }
    },
    "PRICES_CPI": {
        "Name": "Consumer Price Indices",
        "SDMX Query": "OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL",
        "Measurements": {
            "CPI": {
                "Name": "Consumer Price Index - Total",
                "Filters": {
                    "TRANSFORMATION": "G1", 
                    "EXPENDITURE": "_T", 
                    "FREQ": "M", 
                    "METHODOLOGY": "HICP"
                }
            },
            # Furnishings, household equipment and routine household maintenance (CP05) "TRANSFORMATION": "G1", "EXPENDITURE": "CP05", "FREQ": "M", "METHODOLOGY": "HICP"
            "FHE": {
                "Name": "Furnishings, household equipment and routine household maintenance",
                "Filters": {
                    "TRANSFORMATION": "G1", 
                    "EXPENDITURE": "CP05", 
                    "FREQ": "M", 
                    "METHODOLOGY": "HICP"
                }
            },
            # Housing, water, electricity, gas and other fuels (CP04) 
            "HWE": {
                "Name": "Housing, water, electricity, gas and other fuels",
                "Filters": {
                    "TRANSFORMATION": "G1", 
                    "EXPENDITURE": "CP04", 
                    "FREQ": "M", 
                    "METHODOLOGY": "HICP"
                }
            }
        }
    },
    "STLABOUR":{
        "Name": "Short-Term Labour Market Statistics",
        "SDMX Query": "OECD.SDD.TPS,DSD_LFS@DF_IALFS_INDIC",
        "Measurements": {
            "MUE": {
                "Name": "Monthly Unemployment Rate",
                "Filters": {
                    "MEASURE": "UNE_LF_M", 
                    "OBS_STATUS": "A", 
                    "SEX": "_T", 
                    "AGE": "Y_GE15", 
                    "FREQ": "M", 
                    "ADJUSTMENT": "Y"
                }
            }
        }
    },
    "MEI_BTS_COS": {
        "Name": "Business Tendency and Consumer Opinion Surveys",
        "SDMX Query": "OECD.SDD.STES,DSD_STES@DF_BTS",
        "Measurements": {
            "SP": {
                "Name": "Selling Prices",
                "Filters": {
                    "MEASURE": "SP", 
                    "ACTIVITY": "F"
                }
            }
        }
    },
    "MATERIAL_RESOURCES": {
        "Name": "Material Resources",
        "SDMX Query": "OECD.ENV.EPI,DSD_MATERIAL_RESOURCES@DF_MATERIAL_RESOURCES",
        "Measurements": {
            "DMC": {
                "Name": "Domestic Material Consumption",
                "Filters": {
                    "MEASURE": "DMC", 
                    "GROUP": "WOOD", 
                    "UNIT_MEASURE":"T", 
                    "OBS_STATUS": "A"
                }
            },
            "MF": {
                "Name": "Material Footprint",
                "Filters": {
                    "MEASURE": "MF", 
                    "GROUP": "TOT", 
                    "UNIT_MEASURE":"T"
                }
            }
        } 
    }
}
def feature_name(features_df:pd.DataFrame, feature_title):
    filtered = features_df[features_df["Feature"] == feature_title]
    rows = [r for r in filtered.itertuples()]
    return feature_description(rows[0].Source, rows[0].Dataset, rows[0].Measurement, rows[0].Area)[1]
def feature_description(source_key, dataset_key, measurement_key, area_key):
    description = source_key
    short = ""
    if dataset_key in oecd_sources.keys():
        dataset_name = f" {oecd_sources[dataset_key]['Name']}"
        description += dataset_name
        if measurement_key in oecd_sources[dataset_key]["Measurements"].keys():
            measurement_name = f" {oecd_sources[dataset_key]['Measurements'][measurement_key]['Name']}"
            description += measurement_name
            short = measurement_name
        else:
            measurement_name = measurement_key
            description += measurement_key
            short += measurement_key
    else: 
        dataset_name = dataset_key
        description += dataset_key
    if area_key in oecd_categorical_codes["REF_AREA"].keys():
        area_name = f" {oecd_categorical_codes['REF_AREA'][area_key]}"
        description += area_name
        short += area_name
    else:
        area_name = area_key
        description += area_key
        short += area_key
    return description, short, dataset_name, measurement_name, area_name
def feature_descriptions(df:pd.DataFrame):
    result = {}
    for row in df.itertuples():
        description, short, dataset_name, measurement_name, area_name = feature_description(row.Source, row.Dataset, row.Measurement, row.Area)
        if row.Measurement not in result:
            result[row.Measurement] = {
                "Keys": {"Source": row.Source, "Dataset": row.Dataset, "Area": row.Area},
                "Source": row.Source,
                "Name": measurement_name,
                "Dataset": (row.Dataset, dataset_name),
                "Area": []
            }
        result[row.Measurement]["Area"].append((row.Area, area_name))
    for key, info in result.items():
        areas = list_toquery([area_name for _, area_name in info["Area"]], separator=", ")
        result[key]["Description"] = f"{result[key]['Name']} (for {areas})"
    return result
def measure_source_info(measurement_name):
    # Find the corresponding filters and details from the oecd_sources dictionary
    filters = None
    dataset_key = None
    measurement_key = None
    for dataset_key, dataset in oecd_sources.items():
        for measurement_key, measurement in dataset["Measurements"].items():
            if measurement["Name"] == measurement_name:
                filters = measurement["Filters"]
                break
        if filters:
            break
    return dataset_key, measurement_key, filters
def get_from_oecd(oecd_stat_dataset_code):
    return get_from_oecd_sdmx(sdmx_query=oecd_sources[oecd_stat_dataset_code]["SDMX Query"])
def get_from_oecd_sdmx(sdmx_query: str, retries: int = 1, pause: int = 4):
    # pause is 4 by default, because the maximum request frequency of the oecd api is 20/min => 60/20+1=4
    base_url = "https://sdmx.oecd.org/public/rest/data/"
    query_url = f"{base_url}{sdmx_query}/"
    # + ?dimensionAtObservation=AllDimensions

    headers = {
        "Accept": "application/vnd.sdmx.data+csv;version=2.0"
    }
    
    for attempt in range(retries):
        try:
            time.sleep(pause)
            response = requests.get(query_url, headers=headers)
            response.raise_for_status()  # Raise an error for bad responses
            data = pd.read_csv(io.StringIO(response.text))
            return data
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}. Retrying in {pause} seconds...")
    
    print("Failed to retrieve data.")
    return None
    
    # return pd.read_csv(
    #     f"https://stats.oecd.org/SDMX-JSON/data/{sdmx_query}?contentType=csv"
    # )

    # OECD API: https://data.oecd.org/api/sdmx-json-documentation/#d.en.330346
        # from cif import cif
    #data, subjects, measures = cif.createDataFrameFromOECD(countries = ['USA'], dsname = 'MEI_FIN', frequency = 'M')
    # df = pd.read_csv('https://stats.oecd.org/SDMX-JSON/data/MEI_BTS_COS/CSESFT.AUS.BLSA.M/OECD?contentType=csv')
    #df.head()
    # mei_cli = pd.read_json("https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD/?format=jsondata")

def print_uniques(df: pd.DataFrame, columns=None, excluded=None, max_uniques=1500):
    excluded = excluded or []
    for column in columns if columns else df.columns:
        if column not in excluded:
            unique_values = set(df[column])
            if len(unique_values) > max_uniques:
                print(f"Uniques of '{column}': ({len(unique_values)}) Too many to print. Data Type: {df[column].dtype}")
            else:
                print(f"Uniques of '{column}': ({len(unique_values)}) {unique_values}")
def save_oecd_df(df: pd.DataFrame, directory: str="../04 Data/OECD", filename="output"):
    df.to_csv(os.path.join(directory, filename+".csv"))
def filter_oecd_df(df, **filters):
    for key, value in filters.items():
        if value is not None:
            df = df[df[key] == value]
    return df
def process_oecd_data(measurement_name, cut_null_nan=False, only_relevant_regions=False):
    # Find the corresponding filters and details from the dictionary
    dataset_key, measurement_key, filters = measure_source_info(measurement_name)
    
    if filters is None:
        raise ValueError(f"Measurement name '{measurement_name}' not found in OECD sources dictionary.")
    
    df = pd.read_csv(f"../04 Data/OECD/Raw/OECD_{dataset_key}.csv")

    print("Original Data Set")
    print_uniques(df)

    # Add the relevant regions filter if needed
    if only_relevant_regions:
        filters["REF_AREA"] = df["REF_AREA"].isin(oecd_relevant_regions)

    filtered_df = filter_oecd_df(df, **filters)
    
    # Overview on remaining data by printing unique values of each column
    print(f"Data set after filtering for {measurement_name}")
    print_uniques(filtered_df)
    
    print("Check for Duplicated Keys")
    print(check_duplicate_keys(filtered_df,["REF_AREA", "TIME_PERIOD"]))

    # Select the relevant columns
    filtered_df = filtered_df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]]
    
    # Display information about the filtered DataFrame
    filtered_df.info()
    
    # Create a pivot table
    pivoted_df = pd.pivot_table(filtered_df,
                                values='OBS_VALUE',
                                index=['TIME_PERIOD'],
                                columns=['REF_AREA'])
    
    # Reset the 'TIME_PERIOD' index to keep it as a column
    pivoted_df = pivoted_df.reset_index(level='TIME_PERIOD')

    print(pivoted_df)

    na_seqs = sequences_of_na(pivoted_df)
    print(na_seqs)

    # Cut Null/NaN values
    if cut_null_nan:
        pivoted_df = longest_continuous_series(pivoted_df)

        print("Pivoted and cutted off Nulls and NaNs")
        print(pivoted_df)
    
    # Save the processed DataFrame
    save_oecd_df(pivoted_df, directory="../04 Data/OECD", filename=f"OECD_{dataset_key}_{measurement_key}")
    
    return pivoted_df
def sequences_of_na(df):
    na_filter = lambda x: pd.isna(x) or x == "NaN"

    def find_na_sequences(col):
        na_indices = col.apply(na_filter)
        sequences = []
        current_sequence = None
        
        for i, is_na in enumerate(na_indices):
            if is_na:
                if current_sequence is None:
                    current_sequence = [i, i]  # Start a new sequence
                else:
                    current_sequence[1] = i  # Extend the current sequence
            else:
                if current_sequence is not None:
                    sequences.append((current_sequence[0], current_sequence[1], current_sequence[1] - current_sequence[0] + 1))
                    current_sequence = None
        
        # Append the last sequence if it exists
        if current_sequence is not None:
            sequences.append((current_sequence[0], current_sequence[1], current_sequence[1] - current_sequence[0] + 1))
        
        return sequences if sequences else None

    na_sequences = df.apply(find_na_sequences, axis=0)
    na_sequences = na_sequences.apply(lambda x: x if x is not None else [])

    print("Sequences of Null/NaN values")
    return na_sequences
def check_duplicate_keys(df, keys):
    """
    Prüft, ob eine Kombination von Schlüsselattributen mehrere Einträge existieren
    und gibt für jede Schlüssel-Kombination die Anzahl der Einträge zurück.

    :param df: Der DataFrame, der überprüft werden soll.
    :param keys: Liste der Spaltennamen, die die Schlüsselattribute darstellen.
    :return: DataFrame, der die Schlüssel-Kombinationen und die Anzahl der Einträge enthält.
    """
    # Gruppiere den DataFrame nach den Schlüsselattributen und zähle die Einträge
    grouped_df = df.groupby(keys).size().reset_index(name='count')
    
    # Filtere nur die Gruppen, die mehr als einen Eintrag haben
    duplicates = grouped_df[grouped_df['count'] > 1]
    
    return duplicates
def longest_continuous_series(df):
    """
    Findet die längste zusammenhängende Zeitreihe ohne NULL-, NaN-Werte oder den Text "NaN" für alle Spalten und kürzt den DataFrame entsprechend.
    
    :param df: DataFrame mit Zeitreihendaten, indexiert mit einem aufsteigenden Index und einer Zeitangabe im Format yyyyMM.
    :return: DataFrame mit der längsten zusammenhängenden Zeitreihe ohne NULL-, NaN-Werte oder den Text "NaN" für alle Spalten.
    """
    
    na_filter = lambda x: pd.isna(x) or x == "NaN"

    last_null_or_nan_idxs = df.apply(lambda col: (col[::-1].apply(na_filter)).idxmax() if (col.isna() | col.eq("NaN")).any() else 0)
    print("Index of the last Null or NaN value in each column:")
    print(last_null_or_nan_idxs)

    if last_null_or_nan_idxs is not None:
        start_of_longest_series = last_null_or_nan_idxs.max()
    
    if start_of_longest_series is not None:
        # Schneide die Daten bis zum ersten NaN-, NULL-Wert oder "NaN"-Text (exklusiv)
        result_df = df.loc[start_of_longest_series+1:]
    else:
        # Keine NaN-, NULL-Werte oder "NaN"-Text gefunden, gesamte DataFrame ist die längste zusammenhängende Zeitreihe
        result_df = df
    
    return result_df
# Function to expand datasets with quarterly values to monthly values
def expand_quarter(df, time_col):
    quarter_map = {
        'Q1': ['01', '02', '03'],
        'Q2': ['04', '05', '06'],
        'Q3': ['07', '08', '09'],
        'Q4': ['10', '11', '12']
    }
    
    expanded_rows = []
    for _, row in df.iterrows():
        year, quarter = row[time_col].split('-Q')
        for month in quarter_map[f'Q{quarter}']:
            new_time_period = f"{year}-{month}"
            new_row = row.copy()
            new_row[time_col] = new_time_period
            expanded_rows.append(new_row)
    
    return pd.DataFrame(expanded_rows)
def integrate_oecd_data(oecd_sources, oecd_relevant_regions, expand_quarter):
    oecd_df = pd.DataFrame()
    oecd_features = pd.DataFrame(columns=["Feature", "Source", "Dataset", "Measurement", "Description", "Area"])
    
    for dataset, ds_info in oecd_sources.items():
        for measurement, ms_info in ds_info["Measurements"].items():
            # Read
            ms_df = pd.read_csv(f"../04 Data/OECD/OECD_{dataset}_{measurement}.csv")

            # TIME_PERIOD as string
            ms_df['TIME_PERIOD'] = ms_df['TIME_PERIOD'].astype(str)
            
            # Filter for relevant areas
            ms_df = ms_df[["TIME_PERIOD"]+[area for area in oecd_relevant_regions if area in ms_df.columns]]

            # Separate DataFrames
            yearly_df = ms_df[ms_df['TIME_PERIOD'].str.len() == 4]
            quarterly_df = ms_df[ms_df['TIME_PERIOD'].str.contains('-Q')]
            monthly_df = ms_df[(ms_df['TIME_PERIOD'].str.len() == 7) & 
                               (ms_df['TIME_PERIOD'].str.contains('-')) & 
                               (~ms_df['TIME_PERIOD'].str.contains('Q'))]
            ms_df = monthly_df

            # Convert quarterly to monthly
            if not quarterly_df.empty:
                quarterly_df = expand_quarter(quarterly_df, 'TIME_PERIOD')
                # Remove converted quarters that already have monthly values
                quarterly_df = quarterly_df[~quarterly_df['TIME_PERIOD'].isin(monthly_df['TIME_PERIOD'])]
                ms_df = pd.concat([monthly_df, quarterly_df])
            
            # Convert yearly to monthly
            if not yearly_df.empty:
                yearly_df = yearly_df.loc[yearly_df.index.repeat(12)].reset_index(drop=True)
                yearly_df['MONTH'] = (yearly_df.groupby('TIME_PERIOD').cumcount() + 1).astype(str).str.zfill(2)
                yearly_df['TIME_PERIOD'] = yearly_df['TIME_PERIOD'] + '-' + yearly_df['MONTH']
                yearly_df = yearly_df.drop(columns=['MONTH'])
                # Remove converted years that already have monthly values
                yearly_df = yearly_df[~yearly_df['TIME_PERIOD'].isin(ms_df['TIME_PERIOD'])]
                ms_df = pd.concat([ms_df, yearly_df])
            
            # Rename for uniqueness when merged
            areas = [area for area in ms_df.columns if area != "TIME_PERIOD"]
            ms_df = ms_df.rename(columns = {area: f"OECD_{dataset}_{measurement}_{area}" for area in areas})
            
            for area in areas:
                nr = {
                    "Feature": f"OECD_{dataset}_{measurement}_{area}",
                    "Source": "OECD",
                    "Dataset": dataset,
                    "Measurement": measurement,
                    "Description": "",
                    "Area": area
                }
                if nr['Feature'] in ms_df.columns:
                    nr = pd.DataFrame({key: [value] for key, value in nr.items()})
                    oecd_features = pd.concat([oecd_features, nr], ignore_index=True)

            # Merge
            if oecd_df.empty:
                oecd_df = ms_df
            else: 
                oecd_df = pd.merge(oecd_df, ms_df, on="TIME_PERIOD", how="outer")
    
    oecd_df = oecd_df.sort_values('TIME_PERIOD').reset_index(drop=True)

    save_oecd_df(oecd_df, filename="OECD_integrated")
    save_oecd_df(oecd_features, filename="OECD_features")
    
    return oecd_df, oecd_features
def plot_missing_values(df, time_column='TIME_PERIOD'):
    if not isinstance(df.index, pd.DatetimeIndex):
        # Konvertiere die Zeitspalte in das Datetime-Format
        df[time_column] = pd.to_datetime(df[time_column])
        
        # Setze die Zeitspalte als Index
        df.set_index(time_column, inplace=True)
    
    # Erstelle eine leere Figur und Achse
    fig, ax = plt.subplots(figsize=(15, 0.15*df.shape[1]))
    
    # Iteriere über jede Spalte im DataFrame
    for i, column in enumerate(df.columns):
        # Erstelle eine Serie, die 1 ist, wenn der Wert Null oder NaN oder "nan" ist, und 0 sonst
        is_missing = df[column].isna() | (df[column] == "nan")
        
        # Plotten der nicht-null Werte in grün
        ax.plot(df.index[~is_missing], [i] * len(df.index[~is_missing]), 'go', label='Not Null' if i == 0 else "")
        
        # Plotten der null/nan Werte in rot
        ax.plot(df.index[is_missing], [i] * len(df.index[is_missing]), 'ro', label='Null/NaN' if i == 0 else "")
    
    # Setze die y-Achse mit den Spaltennamen
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    
    # Setze die x-Achse als Zeitachse
    ax.xaxis_date()
    
    # Setze die Labels und den Titel
    ax.set_xlabel('Time')
    ax.set_ylabel('Columns')
    ax.set_title('Missing Values in Time Series')
    
    # Füge eine Legende hinzu
    ax.legend(loc='upper right')
    
    # Zeige das Diagramm
    plt.show()
def find_largest_subseries(df: pd.DataFrame, max_consecutive_invalids, time_column='TIME_PERIOD'):
    """
    Find the largest contiguous subsequence in the given dataframe where no column exceeds
    the allowed number of consecutive invalid values (NaN or zero).
    
    Parameters:
    df (pd.DataFrame): The input dataframe with time series data.
    max_consecutive_invalids (int): The maximum number of consecutive invalid values allowed in any column.
    time_column (str): The name of the column containing time indices.
    
    Returns:
    pd.DataFrame: The filtered dataframe with the largest valid subsequence.
    dict: Properties of the resulting dataframe.
    list: Names of columns removed due to too many NaNs at the end.
    list: Names of columns removed during the greedy approach.
    """
    # Ensure the time column is sorted
    df = df.sort_values(by=time_column).reset_index(drop=True)
    
    # Lists to keep track of removed columns
    removed_due_to_nans = []
    removed_greedy = []
    
    # Remove columns with too many NaNs at the end
    filtered_columns = []
    for col in df.columns.difference([time_column]):
        current_streak = 0
        for value in reversed(df[col]):
            if pd.isna(value) or value == "nan":
                current_streak += 1
            else:
                break
        if current_streak <= max_consecutive_invalids:
            filtered_columns.append(col)
        else:
            removed_due_to_nans.append(col)
    
    df = df[[time_column] + filtered_columns]
    
    # Helper function to find the largest valid subsequence for given columns using dynamic programming
    def largest_valid_subsequence(columns):
        n = len(df)
        max_len = 0
        start_idx = 0
        end_idx = 0
        dp = defaultdict(lambda: 0)
        
        for i in range(n):
            valid = True
            for col in columns:
                if pd.isna(df.at[i, col]) or df.at[i, col] == "nan":
                    dp[col] += 1
                else:
                    dp[col] = 0
                
                if dp[col] > max_consecutive_invalids:
                    valid = False
            
            if valid:
                if i - start_idx + 1 > max_len:
                    max_len = i - start_idx + 1
                    end_idx = i
            else:
                start_idx = i + 1
                dp = defaultdict(lambda: 0)
        
        return df.loc[end_idx - max_len + 1:end_idx + 1, [time_column] + list(columns)]
    
    # Greedy approach: Iteratively remove columns with the most invalid values
    best_subsequence = pd.DataFrame()
    max_data_points = 0
    remaining_columns = df.columns.difference([time_column]).tolist()
    
    while remaining_columns:
        # Sort columns by the number of invalid values in descending order
        invalid_counts = {col: df[col].isna().sum() + (df[col] == "nan").sum() for col in remaining_columns}
        sorted_columns = sorted(invalid_counts, key=invalid_counts.get, reverse=True)
        
        found_better = False
        for col in sorted_columns:
            columns_to_check = [c for c in remaining_columns if c != col]
            subsequence = largest_valid_subsequence(columns_to_check)
            data_points = np.sum([subsequence[col].isna().sum() + (subsequence[col] == "nan").sum() for col in subsequence.columns])
            if data_points > max_data_points:
                best_subsequence = subsequence
                max_data_points = data_points
                remaining_columns = columns_to_check
                removed_greedy.append(col)
                found_better = True
                break
        
        if not found_better:
            break
    
    # Calculate properties of the resulting dataframe
    start_date = best_subsequence[time_column].min()
    end_date = best_subsequence[time_column].max()
    length = len(best_subsequence)
    num_features = len(best_subsequence.columns) - 1
    num_datapoints = length * num_features
    
    properties = {
        "Start Date": start_date,
        "End Date": end_date,
        "Length": length,
        "Number of Features": num_features,
        "Max Consecutive NaNs": max_consecutive_invalids,
        "Number of Datapoints": num_datapoints,
        "Removed due to NaNs": removed_due_to_nans,
        "Removed for greedy optimization": removed_greedy
    }
    
    return best_subsequence, properties

# Main Routines
# Main Routine collecting data from oecd api into raw data directory
def collect_oecd_data(directory="../04 Data/OECD/Raw"):
    for dataset, info in oecd_sources.items():
        print(info["Name"]+f" ({dataset})")
        df = get_from_oecd(dataset)
        save_oecd_df(df, directory=directory, filename="OECD_"+dataset)
# Processing raw oecd data into useable datasets (csv)
def process_all_oecd_data():
    measurements = [
        # QNA
        "Gross Domestic Product",
        # MEI_REAL
        "Production in Industry",
        "Production in Manufacturing",
        "Production in Construction",
        # MEI_CLI
        "Composite Leading Indicator",
        "Composite Business Confidence",
        "Composite Consumer Confidence",
        # PRICES_CPI
        "Consumer Price Index - Total",
        "Furnishings, household equipment and routine household maintenance",
        "Housing, water, electricity, gas and other fuels",
        # STLABOUR
        "Monthly Unemployment Rate",
        # MEI_BTS_COS
        "Selling Prices",
        # OECD_MR
        "Domestic Material Consumption",
        "Material Footprint",
        # MEI_FIN
        "Long-Term Interest Rates",
        "Short-Term Interest Rates",
        "Immediate Interest Rates / Interbank Interest Rates",
        "Nominal Interest Rates",
        "Real Effective Exchange Rates - CPI based"
    ]
    for m in measurements:
        process_oecd_data(m)
# Main Routine integrating OECD Data from collected csvs into feature csv for data analysis and modeling data preprocessing
def integrate_and_optimize_oecd_data():
    oecd_df, oecd_features = integrate_oecd_data(oecd_sources, oecd_relevant_regions, expand_quarter)
    if "Unnamed: 0" in oecd_df.columns:
        oecd_df = oecd_df.drop(columns=["Unnamed: 0"])
    oecd_df['TIME_PERIOD'] = pd.to_datetime(oecd_df['TIME_PERIOD'])

    oecd_df_final, properties = find_largest_subseries(oecd_df[(oecd_df['TIME_PERIOD'] > "2000") & (oecd_df['TIME_PERIOD'] < "2024-09")], max_consecutive_invalids=20)

    oecd_df_final= oecd_df_final.reset_index()
    save_oecd_df(oecd_df_final,filename="OECD_final")
    save_oecd_df(oecd_features,filename="OECD_features")
    return oecd_df_final, properties


# Preprocessing HOMAG Data
import pandas as pd
homag_bu_dict = {
    "N": "[+] Nicht zug. PG Konzern(n/e)",
    "O": "[+] Other",
    "CNC": "[+] BU CNC",
    "CSW": "[+] CSW - Consulting and",
    "CES": "[+] BU Construction Elem",
    "Panel": "[+] BU Panel dividing",
    "Edge": "[+] BU Edge processing"
}
def process_date_columns(df):
    df['Y'] = df['Geschäftsjahr'].astype(str)
    df['M'] = df['Buchungsperiode'].apply(lambda x: f"{x:02d}")  # Monat mit führender Null
    df['TIME_PERIOD'] = df['Y'] + '-' + df['M']
    df = df.drop(columns=['Y', 'M'])
    return df
def map_bu_column(df, bu_dict):
    df["PG Konzern"] = df["PG Konzern"].map({value: key for key, value in bu_dict.items()})
    return df
def rename_and_select_columns(df):
    df = df[["TIME_PERIOD", "Land WE*", "PG Konzern", "Nettoumsatz [EUR]"]]
    df = df.rename(columns={"Land WE*": "REF_AREA", "Nettoumsatz [EUR]": "Sales"})
    return df
def convert_sales_column(df):
    df['Sales'] = df['Sales'].str.replace('.', '').astype(float)
    return df
def pivot_sales_data(df):
    monthly_df = df.pivot_table(
        index='TIME_PERIOD', 
        columns='PG Konzern', 
        values='Sales', 
        aggfunc='sum',  
        fill_value=0
    )
    monthly_df['T'] = monthly_df.sum(axis=1)  # Group Total Sales
    monthly_df = monthly_df.reset_index(level='TIME_PERIOD')
    monthly_df.columns.name = None
    return monthly_df
def region_codes(filepath):
    df = pd.read_csv(filepath, sep=";")
    dict = df.set_index("Land WE*")["Land"].to_dict()
    return dict
def preprocess_hg_df(input_filepath, output_filepath, homag_bu_dict):
    # Load data
    hg_df = pd.read_csv(input_filepath, sep=";")
    
    # Display initial data info
    print("Initial Data Info:")
    print(hg_df.info())
    print(hg_df.head(10))
    print_uniques(hg_df)

    # Process data
    hg_df = process_date_columns(hg_df)
    hg_df = rename_and_select_columns(hg_df)
    hg_df = map_bu_column(hg_df, homag_bu_dict)
    hg_df = convert_sales_column(hg_df)
    hg_monthly_df = pivot_sales_data(hg_df)

    # Save processed data
    hg_monthly_df.to_csv(output_filepath, index=False)

    # Display final data info
    print("Processed Data Info:")
    print(hg_monthly_df.info())
    print(hg_monthly_df.head(10))
    print_uniques(hg_monthly_df)

    return hg_monthly_df


# Preprocessing News Data
from newscatcherapi import NewsCatcherApiClient
import requests
import json
from newsapi.newsapi_client import NewsApiClient
dates = [("2024-08-01", "2024-09-01")]
dates_newscatcher = [("30 days ago", "1 day ago")]
# dates = [(start, end) for start, end in [i-timespan(1M), i for i in [datetime.now() - timespan("") for i in range(0, 5*365, 1)]]]
news_topics = {
    "de": [
        "Möbelproduktion",
        "Holzhausproduktion",
        "Möbelindustrie",
        "Holzbauindustrie",
        "Holzhandel",
        "Politische Entwicklungen",
        "Gesetzesänderungen",
        "Subventionsprogramme",
        "Automatisierung"
    ]#,
    #"en": [""]
}
newsapi = NewsApiClient(api_key=os.environ.get("NEWSAPI_API_KEY"))
def get_from_newsapi(newsapi, dates, news_topics, request_date_baseline="from10"):
    #sources = newsapi.get_sources()
    for date in dates:
        (start_date, end_date) = date
        for language, topics in news_topics.items():
            for topic in topics:
                articles = newsapi.get_everything(q=topic,
                                                from_param=start_date,
                                                to=end_date,
                                                language=language,
                                                sort_by="relevancy")

                with open(f'../04 Data/News/Raw NewsAPI/News_{request_date_baseline}_{language}_{topic}.json', 'w') as json_file:
                    json.dump(articles, json_file, indent=4)
newscatcherapi = NewsCatcherApiClient(x_api_key=os.environ.get("NEWSCATCHER_API_KEY"))
def list_toquery(list, separator=" AND "):
    # list_toquery(news_topics["de"], separator=" AND ")
    all = list[0]
    for i, t in enumerate(list):
        if i > 0:
            all += f"{separator}{t}"
    return f"{all}"
def save_response(response, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(response, json_file, indent=4)
def request_newscatcher(query: str, language: str, start: str, end:str):
    url = "https://v3-api.newscatcherapi.com/api/search"
    payload = {
        "q": query,
        "lang": language,
        "sort_by":"relevancy",
        "page_size":"1000",
        #"theme": "Tech",
        "from_": start,
        "to_": end
        #"exclude_duplicates": True
    }
    headers = {
        'x-api-token': os.environ.get("NEWSCATCHER_API_KEY"),
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }

    response = requests.get(url, headers=headers, params=payload)
    return response
def get_from_newscatcher(dates, news_topics, filepath):
    news_raw = {}
    for date in dates:
        (start_date, end_date) = date
        for language, topics in news_topics.items():
            newscatcher_response = request_newscatcher(list_toquery(topics, " OR "), language, start_date, end_date)
            news_raw[start_date] = newscatcher_response.json()
    save_response(news_raw, filepath)


# Google Trends (not used, manually downloaded)
# Keywords in different languages and regions
keywords_by_region = {
    'DE': [
        "Holzbearbeitungsmaschinen", "Möbelindustrie", "Holzhausbau", "Möbel", "Haus", "Küche",
        "Möbelherstellung Maschinen", "Küchenmöbel Produktion", "CNC Holzbearbeitungsmaschinen","Automatisierte Möbelproduktion", "Holzverarbeitungsanlagen", "Nachhaltige Möbelproduktion",
        "Möbeldesign Trends", "Küchenmöbel Trends", "Holzwerkstoffe", "DIY Möbelbau", "Innenausbau Trends", "Smart Home Möbel", "3D-Druck Möbel", "Modulare Holzhäuser", "Holzbau Technologien", "Holzwerkzeug Maschinen", "Holzverarbeitende Industrie"
    ],
    'US': [
        "woodworking machines", "furniture industry", "wooden house construction", "furniture", "house", "kitchen","furniture manufacturing machines", "kitchen furniture production", "CNC woodworking machines", "automated furniture production", "wood processing plants", "sustainable furniture production", "furniture design trends", "kitchen furniture trends", "wood materials", "DIY furniture building", "interior design trends", "smart home furniture", "3D printed furniture", "modular wooden houses", "wood construction technologies", "woodworking tools", "wood processing industry"
    ],
    'FR': [
        "Machines à bois", "Industrie du meuble", "Construction de maisons en bois", "Meubles", "Maison", "Cuisine",
        "Machines de fabrication de meubles", "Production de meubles de cuisine", "Machines CNC pour le travail du bois", "Production automatisée de meubles", "Installations de transformation du bois", "Production de meubles durables", "Tendances du design de meubles", "Tendances des meubles de cuisine", "Matériaux en bois", "Construction de meubles DIY", "Tendances de l'aménagement intérieur", "Meubles intelligents", "Meubles imprimés en 3D", "Maisons en bois modulaires", "Technologies de construction en bois", "Outils de travail du bois", "Industrie de la transformation du bois"
    ],
    'ES': [
        "Máquinas para trabajar la madera", "Industria del mueble", "Construcción de casas de madera", "Muebles", "Casa", "Cocina", "Máquinas de fabricación de muebles", "Producción de muebles de cocina", "Máquinas CNC para trabajar la madera", "Producción automatizada de muebles", "Plantas de procesamiento de madera", "Producción sostenible de muebles", "Tendencias en diseño de muebles", "Tendencias en muebles de cocina", "Materiales de madera", "Construcción de muebles DIY", "Tendencias en diseño de interiores", "Muebles inteligentes", "Muebles impresos en 3D", "Casas de madera modulares", "Tecnologías de construcción en madera", "Herramientas para trabajar la madera", "Industria de procesamiento de madera"
    ],
    'IT': [
        "Macchine per la lavorazione del legno", "Industria del mobile", "Costruzione di case in legno", "Mobili", "Casa", "Cucina", "Macchine per la fabbricazione di mobili", "Produzione di mobili da cucina", "Macchine CNC per la lavorazione del legno", "Produzione automatizzata di mobili", "Impianti di lavorazione del legno", "Produzione sostenibile di mobili", "Tendenze del design dei mobili", "Tendenze dei mobili da cucina", "Materiali in legno", "Costruzione di mobili DIY", "Tendenze di interior design", "Mobili intelligenti", "Mobili stampati in 3D", "Case in legno modulari", "Tecnologie di costruzione in legno", "Utensili per la lavorazione del legno", "Industria della lavorazione del legno"
    ],
    'PT': [
        "Máquinas para trabalhar madeira", "Indústria de móveis", "Construção de casas de madeira", "Móveis", "Casa", "Cozinha", "Máquinas de fabricação de móveis", "Produção de móveis de cozinha", "Máquinas CNC para trabalhar madeira", "Produção automatizada de móveis", "Plantas de processamento de madeira", "Produção sustentável de móveis", "Tendências de design de móveis", "Tendências de móveis de cozinha", "Materiais de madeira", "Construção de móveis DIY", "Tendências de design de interiores", "Móveis inteligentes", "Móveis impressos em 3D", "Casas de madeira modulares", "Tecnologias de construção em madeira", "Ferramentas para trabalhar madeira", "Indústria de processamento de madeira"
    ]
    #, 'CN': [], 'JP': [], 'HN': [], 'TU': [], 'AR': []
}
def serpapi_search_googletrends(name: str, query: str):
    params = {
        "api_key": "d4ef35b301f2509c9c123b41dd633e32d4e59faec824bcfb1b40a64eafecfdef",
        "engine": "google_trends",
        "q": query,
        "data_type": "TIMESERIES",
        "csv": "true",
        "date": "all" #"today 5-y"
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    # Save results as JSON
    with open(f'{name}.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # Save results as csv
    if "csv" in results:
        csv_data = results["csv"]
        # Assuming the csv_data is a string with newline-separated rows
        rows = csv_data.split('\n')
        
        with open(f'{name}.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for row in rows:
                csv_writer.writerow(row.split(','))

    return results


# Stock Movements (Yahoo Finance)
stock_names = {"^GSPC": ("S&P 500", "SP_500"),
                #"^SPG1200": ("S&P GLOBAL 1200", "SP_1200"),
                "^GSPTSE": ("S&P/TSX Composite Index", "SP_TSX_COMPOSITE"),
                "^STOXX50E": ("EUROSTOXX 50", "ESTX50"),
                "^DJI": ("Dow Jones Industrial Average", "DJIA"),
                "^GDAXI": ("DAX PERFORMANCE-INDEX", "GDAX"),
                "^FTSE": ("FTSE 100", "FTSE"),
                "^IXIC": ("NASDAQ Composite", "NASDAQ"),
                "^SP500-20": ("S&P 500 GLOBAL 1200 - Industrials", "SPI"),
                "MSCI": ("MSCI World", "MSCI"),
                "ESIN.F": ("MSCI Europe Industrials", "MSCIEI"),
                "^OSEAX": ("Oslo Børs All-share Index", "OSEAX"),
                "^RUT": ("Russell 2000", "RUSSELL_2000"),
                "^N225": ("Nikkei 225", "NIKKEI_225"),
                "^HSI": ("Hang Seng Index", "HANG_SENG"),
                "^AXJO": ("S&P/ASX 200", "ASX_200")
            }
def resample_to_monthly(dataframe):
    # Resample auf monatliche Daten und Aggregationsfunktion anwenden
    return dataframe[['Open', 'High', 'Low', 'Close']].resample('MS').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        # 'Adj Close': 'last',
        # 'Volume': 'sum'
    })
def load_combine_save(ticker_list, directory, combine_mode='index', resample_monthly=True):
    if combine_mode not in ['index', 'columns']:
        raise ValueError("combine_mode must be either 'index' or 'columns'")
    
    combined_df = pd.DataFrame()

    for ticker in ticker_list:
        file_path = os.path.join(directory, f"{ticker}_alltime.csv")
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        if resample_monthly:
            df = resample_to_monthly(df)
        
        if combine_mode == 'columns':
            df = df.add_prefix(f"{ticker}_")
            df.rename(columns={f"{ticker}_Date": "Date"}, inplace=True)
            df.reset_index(inplace=True)  # Restore 'Date' Column

        if combined_df.empty:
            combined_df = df
        else:
            if combine_mode == 'index':
                df['Ticker'] = ticker  # Ticker-Spalte hinzufügen
                combined_df = pd.concat([combined_df, df], ignore_index=True)
            elif combine_mode == 'columns':
                combined_df = pd.merge(combined_df, df, on='Date', how='outer')
    
    if combine_mode == 'index':
        combined_df.set_index(['Ticker', 'Date'], inplace=True)
    elif combine_mode == 'columns':
        combined_df.set_index('Date', inplace=True)

    return combined_df
def get_ticker_from_combined_df(ticker, dataframe):
    ticker_df = dataframe.xs(ticker, level='Ticker')
    return ticker_df

def collect_stocks(yf_stocks_directory = "../04 Data/Stock Movements Yahoo Finance/Raw", 
                   time_filename_extension = "alltime", stocknames=stock_names):
    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=5*365)
    for stock, stock_info in stock_names.items():
        df = yf.download(stock)
        filename = f"{stock_info[1]}_{time_filename_extension}.csv"
        df.to_csv(f"{yf_stocks_directory}/{stock_info[1]}_{time_filename_extension}.csv")
        #sp500_data.to_json('sp500_data_last_5_years.json', orient='table')
def integrate_stocks_data(stock_names=stock_names):
    stocks_alltime_df = load_combine_save([stock_info[1] for ticker, stock_info in stock_names.items()], "../04 Data/Stock Movements Yahoo Finance/Raw", combine_mode="columns")
    stocks_alltime_df.index.name = "TIME_PERIOD"
    stocks_alltime_df.to_csv("../04 Data/Stock Movements Yahoo Finance/Stocks_integrated.csv")