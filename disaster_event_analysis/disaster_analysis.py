import collections
from itertools import combinations
from argparse import ArgumentParser
from datetime import datetime
import json
from collections import defaultdict
from collections import Counter
import numpy as np
import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt
import math
import operator
import os
import geopy.distance
import heapq
import seaborn as sns
import random
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.iolib.summary2 import summary_col
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import copy
import gzip
import sys
import traceback
# import igraph
# import csv
# import ssl
# import irrCAC
# import jsonlines

# import shap


import urllib.request
import urllib.parse

import socket
import ast
import gc

from utils import News

from pyecharts import options as opts
from pyecharts.charts import Geo
from pyecharts.globals import ChartType, SymbolType, GeoType
from pyecharts.charts import Map
import glob
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.spatial import distance
from scipy.stats import entropy
from statsmodels.iolib.summary import Summary

from utils import unify_url, trans_node_sim,trans_edge_sim, trans_dem_idx_sim, democracy_index_class, distance_class, match_to_distance_class, political_group_list, months, month_len, lang_dict, lang_full_name_dict, LANG_FULL_NAME_MAP, LANG_FAMILY
from utils import lang_list, country_alpha3, top_country_list, french_top_country_list, us_country_list, valid_pair_bound, continent_list

from sortedcontainers import SortedList

# Create a function to return the number of observations
def nobs(x):
    return str(int(x.nobs))

def next_possible_feature(X_npf, y_npf, current_features, ignore_features=[]):
    '''
    This function will loop through each column that isn't in your feature model and
    calculate the r-squared value if it were the next feature added to your model.
    It will display a dataframe with a sorted r-squared value.
    X_npf = X dataframe
    y_npf = y dataframe
    current_features = list of features that are already in your model
    ignore_features = list of unused features we want to skip over
    '''
    #Create an empty dictionary that will be used to store our results
    function_dict = {'predictor': [], 'r-squared':[]}
    #Iterate through every column in X
    shuffle_cols = list(X_npf.columns)
    random.shuffle(shuffle_cols)
    for col in shuffle_cols:
        #But only create a model if the feature isn't already selected or ignored
        if col not in (current_features+ignore_features):
            #Create a dataframe called function_X with our current features + 1
            selected_X = X_npf[current_features + [col]]
            #Fit a model for our target and our selected columns
            model = sm.OLS(y_npf, sm.add_constant(selected_X)).fit()
            #Predict what  our target would be for our selected columns
            y_preds = model.predict(sm.add_constant(selected_X))
            #Add the column name to our dictionary
            function_dict['predictor'].append(col)
            #Calculate the r-squared value between the target and predicted target
            r2 = np.corrcoef(y_npf, y_preds)[0, 1]**2
            #Add the r-squared value to our dictionary
            function_dict['r-squared'].append(r2)
    # Once it's iterated through every column, turn our dict into a sorted DataFrame
    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)

    return function_df.iloc[0]


def capitalize_except(words):
    exceptions = ["the", "of"]

    # Split the string into words
    word_list = words.split()

    # Process each word in the list
    for i, word in enumerate(word_list):
        # If the word is not in the exceptions list, or it's the first word, capitalize it
        if word.lower() not in exceptions or i == 0:
            word_list[i] = word.capitalize()
        else:
            word_list[i] = word.lower()

    # Join the words back into a string
    return ' '.join(word_list)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-dsv", "--datasave", dest="data_save",
                        default=True, action='store_true',
                        help="save the data into csv for storage.")
    parser.add_argument("-dt", "--data", dest="data",
                        default="conflict_fr", type=str,
                        help="can be chosen from disaster_en, disaster_es, disaster_fr, disaster_all, conflict_en, conflict_es, conflict_fr, conflict_all.")
    parser.add_argument("-cf", "--count_format", dest="count_format",
                        default="aver_total", type=str,
                        help="can be chosen from total, aver_total. The difference is to compute the total number of news articles per country")
    parser.add_argument("-bc", "--binary_class", dest="binary_class",
                        default=False, action='store_true',
                        help="convert the news article count as binary class, i.e., either 0 or 1")
    parser.add_argument("-opt", "--option", dest="option",
                            default="rm_pair_mix_factor_lang_summary", type=str,
                            help="can be chosen from gm_country, rm_pair_summary, rm_pair_lang_summary, rm_pair_mix_factor_us_summary, rm_pair_mix_factor_lang_summary, rm_pair_country_summary, rm_pair_country, rm_pair, rm_country, plot.")

    args = parser.parse_args()


    disaster_country_stats = defaultdict(lambda: defaultdict(int))
    disaster_country_pair_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # count news number of each disaster/conflict
    disaster_country_pair_stats_total = defaultdict(lambda: defaultdict(int))
    disaster_country_pair_stats_aver_total = defaultdict(lambda: defaultdict(float))
    disaster_country_pair_stats_aver_list = defaultdict(lambda: defaultdict(list))

    # count death in the disaster/conflict
    disaster_country_pair_stats_death_total = defaultdict(lambda: defaultdict(int))
    disaster_country_pair_stats_aver_death_total = defaultdict(lambda: defaultdict(float))
    disaster_country_pair_stats_aver_death_list = defaultdict(lambda: defaultdict(list))
    data_route = "../../mediacloud/ner_art_sampling/"

    # choose dataset
    if args.data == "disaster_en":
        source_route = "data/disaster_news_count_en/"
    elif args.data == "disaster_es":
        source_route = "data/disaster_news_count_es/"
    elif args.data == "disaster_fr":
        source_route = "data/disaster_news_count_fr/"
    elif args.data == "disaster_all":
        source_route = "data/disaster_*/"
    elif args.data == "conflict_en":
        source_route = "data/conflict_news_count_en/"
    elif args.data == "conflict_es":
        source_route = "data/conflict_news_count_es/"
    elif args.data == "conflict_fr":
        source_route = "data/conflict_news_count_fr/"
    elif args.data == "conflict_all":
        source_route = "data/conflict_*/"

    #
    if "disaster" in args.data:
        death_source = "data/emdat-disaster-instance-info.json"
    elif "conflict" in args.data:
        death_source = "data/gtd-disaster-instance-info.json"

    # for disaster either "total deaths" or "total affected"
    # for conflict "num fatalities" or "num injured"
    with open(death_source) as json_file:
        death_dict = json.load(json_file)
    death_valid_count_dict = defaultdict(int)
    death_count_event_dict = {}
    max_death_num = 0
    for event in death_dict:
        for cur_type in death_dict[event]:
            if int(death_dict[event][cur_type]) >= 0:
                death_valid_count_dict[cur_type] += 1
        if death_source == "data/emdat-disaster-instance-info.json":
            death_count_event_dict[event] = int(death_dict[event]["total deaths"])
        elif death_source == "data/gtd-disaster-instance-info.json":
            death_count_event_dict[event] = int(death_dict[event]["num fatalities"])

        max_death_num = max(max_death_num, death_count_event_dict[event])
    print(death_valid_count_dict)
    print("max_death:", max_death_num)

    event_count = 0
    # compute the average
    valid_death_total = 0
    valid_death_count = 0
    for cur_death_key in death_count_event_dict:
        if death_count_event_dict[cur_death_key] >= 0:
            valid_death_total += death_count_event_dict[cur_death_key]
            valid_death_count += 1

    for cur_death_key in death_count_event_dict:
        if death_count_event_dict[cur_death_key] >= 0:
            event_count += 1
        else:
            # death_count_event_dict[cur_death_key] = 0
            death_count_event_dict[cur_death_key] = int(valid_death_total/valid_death_count)
            print("missing death number:", cur_death_key)
            event_count += 1
    print("event_count: ", event_count)

    output_route = 'csv_output/' + args.data + '.csv'
    output_aver_route = 'csv_output/' + args.data + '_aver.csv'
    top_event_route = 'csv_output/' + args.data + '_top_event.csv'
    release_route = 'csv_output/' + args.data + '_release.csv'
    country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")
    country_alpha3_code = country_geography_list["Alpha-3 code"].to_list()
    country_full_name = country_geography_list["Country"].to_list()
    country_name2alpha3 = {}
    for i in range(len(country_full_name)):
        country_name2alpha3[country_full_name[i]] = country_alpha3_code[i]

    event_art_count_dict = defaultdict(int)
    disaster_country_count = defaultdict(int)
    full_report_dict = {}

    disaster_country_event_class_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    files = list(glob.glob( source_route + '*.jsonl'))
    for file in files:
        cur_split = file.split('/')[2].split('-')
        cur_disaster_type, cur_country_full_name = cur_split[0], cur_split[1]
        if cur_country_full_name == "united states of america":
            cur_country_full_name = "united states"

        # if "disaster" in args.data:
        #     cur_death_key = cur_split[0] + "-" + cur_split[1] + "-" + cur_split[2] + "-" + cur_split[3] + "-" + cur_split[4]
        # elif "conflict" in args.data:
        #     cur_death_key = cur_split[0] + "-" + cur_split[2] + "-" + cur_split[3] + "-" + cur_split[4] + "-" + cur_split[1]

        if "disaster" in args.data:
            cur_death_key = cur_split[0] + "-" + cur_country_full_name + "-" + cur_split[2] + "-" + cur_split[3] + "-" + cur_split[4]
        elif "conflict" in args.data:
            cur_death_key = cur_split[0] + "-" + cur_split[2] + "-" + cur_split[3] + "-" + cur_split[4] + "-" + cur_country_full_name

        # filtered out George floyd event
        if cur_death_key == "attack-2020-05-29-united states":
            continue

        cur_country_full_name = capitalize_except(cur_country_full_name)

        if cur_country_full_name == "United States":
            print(cur_death_key)

        event_art_count_dict[cur_death_key] += len(pd.read_json(path_or_buf=file, lines=True))

        try:
            cur_report_alpha3_list = pd.read_json(path_or_buf=file, lines=True)['country']
        except:
            continue

        if (cur_country_full_name in country_name2alpha3):
            disaster_country_count[country_name2alpha3[cur_country_full_name]] += 1
            cur_report_alpha3_counter = Counter(cur_report_alpha3_list)

            for cur_alpha3_report in cur_report_alpha3_list:
                disaster_country_stats[cur_disaster_type][country_name2alpha3[cur_country_full_name]] += 1
                disaster_country_pair_stats[cur_disaster_type][country_name2alpha3[cur_country_full_name]][cur_alpha3_report] += 1
                disaster_country_pair_stats_total[country_name2alpha3[cur_country_full_name]][cur_alpha3_report] += 1
                disaster_country_pair_stats_aver_list[country_name2alpha3[cur_country_full_name]][cur_alpha3_report].append(cur_report_alpha3_counter[cur_alpha3_report])

                disaster_country_event_class_dict[country_name2alpha3[cur_country_full_name]][cur_alpha3_report][cur_disaster_type] += 1

                if (cur_death_key in death_count_event_dict) and death_count_event_dict[cur_death_key] >= 0:
                    disaster_country_pair_stats_death_total[country_name2alpha3[cur_country_full_name]][cur_alpha3_report] += death_count_event_dict[cur_death_key]
                    disaster_country_pair_stats_aver_death_list[country_name2alpha3[cur_country_full_name]][cur_alpha3_report].append(death_count_event_dict[cur_death_key])
                else:
                    disaster_country_pair_stats_death_total[country_name2alpha3[cur_country_full_name]][cur_alpha3_report] += int(valid_death_total/valid_death_count)
                    disaster_country_pair_stats_aver_death_list[country_name2alpha3[cur_country_full_name]][cur_alpha3_report].append(int(valid_death_total/valid_death_count))

                if cur_alpha3_report not in full_report_dict:
                    full_report_dict[cur_alpha3_report] = 0

    print("total report country number:", len(full_report_dict))
    print("disaster count in oceiania:", disaster_country_count["AUS"])
    print("disaster count in oceiania:", disaster_country_count["NZL"])
    print("average disaster count: ", np.mean(list(disaster_country_count.values())))
    # count articles of each disaster
    for cur_disaster_type in disaster_country_pair_stats:
        cur_count = 0
        for cur_country1 in disaster_country_pair_stats[cur_disaster_type]:
            cur_count += sum(disaster_country_pair_stats[cur_disaster_type][cur_country1].values())
        print(cur_disaster_type, cur_count)

    top_event_data = []
    for k, v in Counter(event_art_count_dict).most_common():
    # for k, v in Counter(event_art_count_dict).most_common(10):
        try:
            top_event_data.append({'event': k, 'article number': v, 'death': death_count_event_dict[k]})
            print('%s: %i, death %i' % (k, v, death_count_event_dict[k]))
        except:
            top_event_data.append({'event': k, 'article number': v, 'death': 'N/A'})
            print('%s: %i, death ...' % (k, v))

        top_event_df = pd.DataFrame(top_event_data)

        top_event_df.to_csv(top_event_route)

    # transformed the count to log scale for more fair comparison
    for cur_disaster_type in disaster_country_stats:
        for cur_alpha3 in disaster_country_stats[cur_disaster_type]:
            disaster_country_stats[cur_disaster_type][cur_alpha3] = np.log(1+disaster_country_stats[cur_disaster_type][cur_alpha3])

    for cur_disaster_type in disaster_country_pair_stats:
        for cur_alpha3_disaster in disaster_country_pair_stats[cur_disaster_type]:
            for cur_alpha3_report in disaster_country_pair_stats[cur_disaster_type][cur_alpha3_disaster]:
                disaster_country_pair_stats[cur_disaster_type][cur_alpha3_disaster][cur_alpha3_report] = np.log(1+disaster_country_pair_stats[cur_disaster_type][cur_alpha3_disaster][cur_alpha3_report])

    for cur_alpha3_disaster in disaster_country_pair_stats_total:
        for cur_alpha3_report in disaster_country_pair_stats_total[cur_alpha3_disaster]:
            disaster_country_pair_stats_total[cur_alpha3_disaster][cur_alpha3_report] = np.log(1+disaster_country_pair_stats_total[cur_alpha3_disaster][cur_alpha3_report])
            disaster_country_pair_stats_death_total[cur_alpha3_disaster][cur_alpha3_report] = np.log(1+disaster_country_pair_stats_death_total[cur_alpha3_disaster][cur_alpha3_report])
    for cur_alpha3_disaster in disaster_country_pair_stats_aver_list:
        for cur_alpha3_report in disaster_country_pair_stats_aver_list[cur_alpha3_disaster]:
            disaster_country_pair_stats_aver_total[cur_alpha3_disaster][cur_alpha3_report] = np.mean([np.log(1+a) for a in disaster_country_pair_stats_aver_list[cur_alpha3_disaster][cur_alpha3_report]])
            disaster_country_pair_stats_aver_death_total[cur_alpha3_disaster][cur_alpha3_report] = np.mean([np.log(1+a) for a in disaster_country_pair_stats_aver_death_list[cur_alpha3_disaster][cur_alpha3_report]])
    # choose outcome in the regression analysis
    if args.count_format == "total":
        disaster_country_pair_stats_total = disaster_country_pair_stats_total
        disaster_country_pair_stats_death_total = disaster_country_pair_stats_death_total
    elif args.count_format == "aver_total":
        disaster_country_pair_stats_total = disaster_country_pair_stats_aver_total
        disaster_country_pair_stats_death_total = disaster_country_pair_stats_aver_death_total

    if args.binary_class:
        for key, subdict in disaster_country_pair_stats_total.items():
            for subkey, value in subdict.items():
                subdict[subkey] = 1 if value > 0 else 0

    if args.data_save:
        # save metadata for visualization
        flat_data = []
        for row_key, nested_dict in disaster_country_pair_stats_total.items():
            for col_key, value in nested_dict.items():
                flat_data.append({'event_country': row_key, 'report_country': col_key, 'art_num': value})
        output_df = pd.DataFrame(flat_data)

        output_df.to_csv(output_route)

        flat_aver_data = []
        for row_key, nested_dict in disaster_country_pair_stats_total.items():
            for col_key, value in nested_dict.items():
                flat_aver_data.append({'event_country': row_key, 'report_country': col_key, 'art_num': value})
        output_aver_df = pd.DataFrame(flat_aver_data)

        output_aver_df.to_csv(output_aver_route)


    ''' why??? '''
    # insert disaster_country_pair_stats_total into disaster_country_pair_stats for easier coding
    disaster_country_pair_stats['total'] = disaster_country_pair_stats_total

    # f = open("output-table-earthquakeevent-pubcountry-frequency-12.27.jsonl")
    #
    # country_disaster_count = defaultdict(int)
    # country_disaster_pairwise_count = defaultdict(lambda: defaultdict(int))
    #
    # line = f.readline()
    # while line:
    #     cut = line.split(" ")
    #     if len(cut) == 3 and cut[1] == '----':
    #         country_disaster_count[cut[0]] += int(cut[2].replace("\n",""))
    #         country_disaster_pairwise_count[reported_country_alpha3][cut[0]] += int(cut[2].replace("\n",""))
    #
    #     else:
    #         cut2 = line.split("earthquake-")
    #         if len(cut2)>1:
    #             reported_country = cut2[1].split("-2020")[0].replace('-', " ").title()
    #             if reported_country in country_alpha3:
    #                 reported_country_alpha3 = country_alpha3[reported_country]
    #             else:
    #                 reported_country_alpha3 = "NaN"
    #     # prev_line = line
    #     line = f.readline()
    #
    # f.close()

    # global map of us media focus on other all countries
    if args.option == "gm_country":
        cur_lang = args.data[-2:]

        '''load democracy index'''
        country_democracy_index_list = pd.read_csv("../../mediacloud/ner_art_sampling/bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list["Country"] = country_democracy_index_list["Country"].apply(
            lambda x: "United States" if x == "US" else x)
        country_democracy_index_list["Country"] = country_democracy_index_list["Country"].apply(
            lambda x: "United Kingdom" if x == "UK" else x)

        country_unitary_state_list = pd.read_csv("../../mediacloud/ner_art_sampling/country_info/country_unitary_state.csv")

        country_alpha3_full_name_list = pd.merge(country_democracy_index_list, country_geography_list, on='Country')
        country_alpha3_full_name_list = pd.merge(country_alpha3_full_name_list, country_unitary_state_list, on='Country')

        ''' geographic visualization '''
        country_alpha3_list = country_alpha3_full_name_list['Alpha-3 code'].to_list()
        country_full_name_list = country_alpha3_full_name_list['Country'].to_list()
        country_alpha3_full_name_dict = {country_alpha3_list[i]:country_full_name_list[i] for i in range(len(country_alpha3_list))}
        country_full_name_alpha3_dict = {country_full_name_list[i]: country_alpha3_list[i] for i in range(len(country_alpha3_list))}

        cur_lang_top_country_list = [i for i in top_country_list if cur_lang in top_country_list[i]]

        # cur_data = pd.read_csv(output_route)
        cur_data = pd.read_csv(output_aver_route)
        cur_data = cur_data[cur_data['report_country'].isin(cur_lang_top_country_list)]

        cur_data_event_country_full_name = [country_alpha3_full_name_dict[c] if c in country_alpha3_full_name_dict else "" for c in cur_data['event_country'].to_list()]
        cur_data_art_num = cur_data['art_num'].to_list()
        cur_data_total_art_num = defaultdict(int)

        cur_data_event_total_country_full_name = cur_data['event_country'].unique().tolist()
        cur_data_event_total_country_full_name = [country_alpha3_full_name_dict[c] if c in country_alpha3_full_name_dict else "" for c in cur_data_event_total_country_full_name]
        # cur_data_total_art_num = cur_data.groupby('event_country')['art_num'].sum().tolist()
        cur_data_event_country_alpha3 = [country_full_name_alpha3_dict[c] if c != "" else "" for c in cur_data_event_total_country_full_name]
        for index, row in cur_data.iterrows():
            cur_data_total_art_num[row['event_country']] += row['art_num']
        cur_data_total_art_num_list = [cur_data_total_art_num[alpha3] if alpha3 != "" else 1 for alpha3 in cur_data_event_country_alpha3]
        # plot global articles of each country
        a = cur_data_event_total_country_full_name
        b = [np.log(c) for c in cur_data_total_art_num_list]

        t = {a[i]:b[i] for i in range(len(a))}
        # a = cur_data_event_country_full_name
        # b = [c for c in cur_data_art_num]
        print("total articles we covered: ", sum(cur_data_art_num))
        geo = (
            Map(init_opts=opts.InitOpts(bg_color="#FFFFFF", theme='essos', width="1500px", height='900px'))  # 图表大小
                .add("", [list(z) for z in zip(a, b)], "world", is_map_symbol_show=False)
                .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 标签不显示(国家名称不显示)
                .set_global_opts(
                title_opts=opts.TitleOpts(title="Article numbers of each country", subtitle='article numbers'),
                # 主标题与副标题名称
                visualmap_opts=opts.VisualMapOpts(min_=0, max_=4.6), # 值映射最大值
            )
        )
        geo.render("geo_art.html")

    if args.option == "rm_pair":

        country_geography_list = pd.read_csv(data_route+ "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route+ "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',on='Country')




        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route+ "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(lambda x: x["borders"].split(",") if isinstance(x["borders"],str) else [], axis=1)
        country_neighbors = country_neighbors.drop(["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left',on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route+ "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(data_route+ "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left', on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route+ "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list, how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route+ "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left', on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route+ "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route+ "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route+ "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0",""), axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route+ f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x:group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass
        # investment flow
        country_pair_investment = pd.read_csv(data_route+ "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
        country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in country_pair_investment.keys()}
        country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_dict["2018"][country_idx]
                country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_investment_dict["2018"][country_idx]
            except:
                pass
        # trade flow
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += \
                country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                country_pair_trade_dict["Value"][country_idx]
            except:
                pass

        # immgration flow
        country_pair_immgration = pd.read_csv(data_route+ "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in country_pair_immgration.keys()}
        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_pair_immgration["Country"])):
            for country2_idx in range(len(country_pair_immgration["Country"])):
                try:
                    cur_country1 = country_pair_immgration_dict["Country"][country1_idx]
                    cur_country2 = country_pair_immgration_dict["Country"][country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass

        # country_syn_dict = country_syn_dict.apply(lambda x: x if len(x['Official language'].intersect(lang_list))>0, axis=1)

        # rm_pairs_dict = {key: rm_pairs[key].to_list() for key in rm_pairs.keys()}

        # building regression model
        for cur_disaster_type in disaster_country_pair_stats:

            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if (cur_alpha3_1 not in disaster_country_pair_stats[cur_disaster_type]) or (cur_alpha3_2 not in disaster_country_pair_stats[cur_disaster_type][cur_alpha3_1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats[cur_disaster_type][cur_alpha3_1][cur_alpha3_2]

                        # cur_lang1 = country_syn_dict["Official language"][cur_idx1]
                        # cur_lang2 = country_syn_dict["Official language"][cur_idx1]
                        # same_lang = cur_lang1.intersect(cur_lang2)
                        # same_lang_vec = [0 for i in range(len(lang_list))]
                        # if len(same_lang) > 0:
                        #     country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_lang"] = lang_dict[cur_alpha3_1][cur_alpha3_2]
                        #     for lang in same_lang:
                        #         if lang in lang_list:
                        #             same_lang_vec[lang_list.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                        # Region1 similarity
                        if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                        # language
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion') or (
                                country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(3)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                if gdp_idx_sign1 == 0:
                                    gdp_vec[2] = 1
                                else:
                                    gdp_vec[3] = 1

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(4)]

                        gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0

                                if gini_index_idx_sign1 == 0:
                                    gini_index_vec[2] = 1
                                else:
                                    gini_index_vec[3] = 1
                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(4)]

                        dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0

                                if dem_idx_sign1 == 0:
                                    dem_idx_vec[2] = 1
                                else:
                                    dem_idx_vec[3] = 1
                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(4)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country1'] = country_alpha3_geography[cur_alpha3_1]["country_full_name"]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country2'] = country_alpha3_geography[cur_alpha3_2]["country_full_name"]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment"] = country_pair_investment_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            model_sets = []
            for search_metric in ['vif','aic']:
                inter_country_train_data = []
                inter_country_train_label = []
                fitlered_country_pair_num = 0
                fitlered_sampling_pair_num = 0

                country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                    for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                        cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                        cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                        cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                        cur_country_pair_sim = cur_country_pair["sim"]
                        cur_country_pair_same_lang = cur_country_pair["same_lang"]
                        cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                        cur_country_pair.pop('simple_lang_vec')
                        cur_country_pair.pop('lang_vec')
                        cur_country_pair.pop('sim')
                        cur_country_pair.pop('same_lang')
                        cur_country_pair.pop('same_spec_lang')


                        if np.isnan(cur_country_pair['border']):
                            continue
                        if np.isnan(cur_country_pair['diplomatic']):
                            continue
                        if np.isnan(cur_country_pair['investment']):
                            continue
                        if np.isnan(cur_country_pair['trade']):
                            continue
                        if np.isnan(cur_country_pair['immgration']):
                            continue

                        for attr in cur_country_pair.values():
                            if isinstance(attr, int) or isinstance(attr, float):
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                            # if isinstance(attr, list) and len(attr) == 1:
                            if isinstance(attr, list):
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                        # feature_combo_list = []
                        # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                        #     temp_list = []
                        #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                        #         temp_list.append(c)
                        #     feature_combo_list.extend(temp_list)

                        try:

                            cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + cur_country_pair_simple_lang_vec + cur_country_pair_lang_vec
                            cur_train_data_row = cur_train_data_row[1:]
                            inter_country_train_data.append(cur_train_data_row)

                            inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                        except:
                            fitlered_country_pair_num += 1
                print("fitlered_country_pair_num: ", fitlered_country_pair_num)

                # normalization
                minmax = MinMaxScaler()
                inter_country_train_data = minmax.fit_transform(inter_country_train_data)

                print("country pairs total number:", len(inter_country_train_label))

                inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
                inter_country_train_data_df.columns = ["Neighbor", "Geography distance","Continent sim", "Region1 sim",\
                                                    'Same political group','Both in NATO', 'Both in EU', 'Both in BRICS',\
                                                   'Across NATO-EU groups', 'Across NATO-BRICS groups', "Across EU-BRICS groups",\
                                                   "Same political system type", "Both republic", "Both federalism", \
                                                   "Neither republic nor federalism", "Republic and federalism", 'Republic and other', 'Federalism and other', \
                                                   "Same GDP category", "GDP(low-low)", "GDP(high-high)", "GDP(low->high)", "GDP(high->low)",\
                                                   "Same Gini Index category", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low->high)", "Gini Index(high->low)",\
                                                   "Same Democracy index category", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low->high)", "Democracy index(high->low)",\
                                                   "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                                   "Same language", "Speaking English", "Speaking German", "Speaking Spanish", "Speaking Polish", "Speaking French", \
                                                   "Speaking Chinese", "Speaking Arabic", "Speaking Turkish", "Speaking Italian", "Speaking Russian"]


                # 0 is the pair number, 1 is the average similarity
                # y = inter_country_train_data_df[1]
                y = inter_country_train_label
                X = inter_country_train_data_df

                X_feature_names = X.columns.values.tolist()
                valid_X_feature_names = []
                # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                for cur_feature in X_feature_names:
                    multivalue_sign = 0

                    cur_feature_values = X[cur_feature]
                    cmp_value = cur_feature_values[0]
                    for cur_feature_value in cur_feature_values:
                        if cur_feature_value != cmp_value:
                            multivalue_sign = 1
                            break
                    if multivalue_sign == 1:
                        valid_X_feature_names.append(cur_feature)

                X = X[valid_X_feature_names]

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

                # Create an empty dictionary that will be used to store our results
                function_dict = {'predictor': [], 'r-squared': []}
                # Iterate through every column in X
                cols = list(X.columns)
                for col in cols:
                    # Create a dataframe called selected_X with only the 1 column
                    selected_X = X[[col]]
                    # Fit a model for our target and our selected column
                    model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                    # Predict what our target would be for our model
                    y_preds = model.predict(sm.add_constant(selected_X))
                    # Add the column name to our dictionary
                    function_dict['predictor'].append(col)
                    # Calculate the r-squared value between the target and predicted target
                    r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                    # Add the r-squared value to our dictionary
                    function_dict['r-squared'].append(r2)

                # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
                function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
                # Display only the top 5 predictors

                if search_metric == "vif":
                    selected_features = [function_df['predictor'].iat[0]]
                    features_to_ignore = []

                    # Since our function's ignore_features list is already empty, we don't need to
                    # include our features_to_ignore list.
                    while len(selected_features) + len(features_to_ignore) < len(cols):
                        next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                                 ignore_features=features_to_ignore)[0]
                        # check vif score
                        vif_factor = 5
                        temp_selected_features = selected_features + [next_feature]
                        temp_X = X[temp_selected_features]
                        temp_vif = pd.DataFrame()
                        temp_vif["features"] = temp_X.columns
                        temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                        cur_vif = temp_vif["VIF"].iat[-1]
                        if cur_vif <= vif_factor:
                            selected_features = temp_selected_features
                        else:
                            features_to_ignore.append(next_feature)
                elif search_metric == "aic":
                    selected_features = [function_df['predictor'].iat[0]]
                    rest_features = []
                    for cur_feature in valid_X_feature_names:
                        if cur_feature not in selected_features:
                            rest_features.append(cur_feature)

                    best_aic = 10000000
                    search_max_time = 10000
                    search_time = 0
                    while len(selected_features) < len(cols) or search_time >= search_max_time:
                        # if there is no change in this turn then meaning no feature is selected.
                        # Should also stop search in this case
                        change_sign = 0
                        temp_feature_sets = [selected_features+[temp_feature] for temp_feature in rest_features]
                        for temp_feature_set in temp_feature_sets:
                            temp_X = X[temp_feature_set]
                            temp_model = sm.OLS(y, sm.add_constant(temp_X)).fit()
                            if temp_model.aic < best_aic:
                                best_aic = temp_model.aic
                                selected_features = temp_feature_set
                                change_sign = 1
                        if change_sign == 0:
                            break
                        search_time += 1


                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                model_sets.append(model)

            model_sets_res = summary_col(model_sets, stars=True)
            print("disaster:", cur_disaster_type)
            cur_save_folder = f"results/disaster_country_pair/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_disaster_type + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())

    if args.option == "rm_pair_country":

        country_geography_list = pd.read_csv(data_route+ "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route+ "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',on='Country')




        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route+ "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(lambda x: x["borders"].split(",") if isinstance(x["borders"],str) else [], axis=1)
        country_neighbors = country_neighbors.drop(["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left',on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route+ "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(data_route+ "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left', on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route+ "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list, how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route+ "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left', on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route+ "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route+ "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route+ "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0",""), axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route+ f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x:group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass
        # investment flow
        country_pair_investment = pd.read_csv(data_route+ "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
        country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in country_pair_investment.keys()}
        country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_dict["2018"][country_idx]
                country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_investment_dict["2018"][country_idx]
            except:
                pass
        # trade flow
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += \
                country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                country_pair_trade_dict["Value"][country_idx]
            except:
                pass

        # immgration flow
        country_pair_immgration = pd.read_csv(data_route+ "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in country_pair_immgration.keys()}
        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_pair_immgration["Country"])):
            for country2_idx in range(len(country_pair_immgration["Country"])):
                try:
                    cur_country1 = country_pair_immgration_dict["Country"][country1_idx]
                    cur_country2 = country_pair_immgration_dict["Country"][country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass

        # country_syn_dict = country_syn_dict.apply(lambda x: x if len(x['Official language'].intersect(lang_list))>0, axis=1)

        # rm_pairs_dict = {key: rm_pairs[key].to_list() for key in rm_pairs.keys()}

        # building regression model
        # for select_country in french_top_country_list:
        for select_country in top_country_list:
            for cur_disaster_type in disaster_country_pair_stats:

                for i in range(len(country_syn_list)):
                    for j in range(len(country_syn_list)):
                        if i == j:
                            continue
                        else:
                            cur_idx1 = i
                            cur_idx2 = j

                            cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                            cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                            '''limit to usa news converage to global disasters'''
                            if cur_alpha3_2 != select_country:
                                continue

                            if (cur_alpha3_1 not in disaster_country_pair_stats[cur_disaster_type]) or (cur_alpha3_2 not in disaster_country_pair_stats[cur_disaster_type][cur_alpha3_1]):
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats[cur_disaster_type][cur_alpha3_1][cur_alpha3_2]


                            # cur_lang1 = country_syn_dict["Official language"][cur_idx1]
                            # cur_lang2 = country_syn_dict["Official language"][cur_idx1]
                            # same_lang = cur_lang1.intersect(cur_lang2)
                            # same_lang_vec = [0 for i in range(len(lang_list))]
                            # if len(same_lang) > 0:
                            #     country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["same_lang"] = lang_dict[cur_alpha3_1][cur_alpha3_2]
                            #     for lang in same_lang:
                            #         if lang in lang_list:
                            #             same_lang_vec[lang_list.index(lang)] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                            if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                            # continent similarity
                            if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                            # Region1 similarity
                            if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                            # language
                            try:
                                common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                            except:
                                common_official_langs = []
                            try:
                                common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                            except:
                                common_official_langs_family = []

                            simple_lang_vector = [0 for i in range(1)]
                            if common_official_langs != []:
                                simple_lang_vector[0] = 1
                            # elif common_official_langs_family != []:
                            #     simple_lang_vector[1] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                            lang_vector = [0 for i in range(10)]
                            if common_official_langs != []:
                                lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                                for lang in common_official_langs:
                                    if lang in lang_full_name_map_keys:
                                        lang_vector[lang_full_name_map_keys.index(lang)] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                            # political group categories, 4 kinds
                            common_group = []
                            for group in ['nato', 'eunion', 'brics']:
                                if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                    common_group.append(group)

                            across_group = []
                            if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                                cur_idx2] == 'eunion') or (
                                    country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                                cur_idx2] == 'nato'):
                                across_group.append('nato-eunion')
                            if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                                cur_idx2] == 'brics') or (
                                    country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                                cur_idx2] == 'nato'):
                                across_group.append('nato-brics')
                            if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                                cur_idx2] == 'brics') or (
                                    country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                                cur_idx2] == 'eunion'):
                                across_group.append('eunion-brics')

                            simple_political_group_vector = [0]
                            if common_group != []:
                                simple_political_group_vector[0] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                            '''political group featrues'''
                            political_group_vector = [0 for i in range(3)]
                            if 'nato' in common_group:
                                political_group_vector[0] = 1
                            elif 'eunion' in common_group:
                                political_group_vector[1] = 1
                            elif 'brics' in common_group:
                                political_group_vector[2] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                                "political_group_vec"] = political_group_vector

                            across_group_vector = [0 for i in range(3)]
                            if 'nato-eunion' in across_group:
                                across_group_vector[0] = 1
                            if 'nato-brics' in across_group:
                                across_group_vector[1] = 1
                            if 'eunion-brics' in across_group:
                                across_group_vector[2] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                            # unitary type combination categories, 6 kinds
                            simple_unitary_vector = [0]
                            if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                                simple_unitary_vector[0] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                                "simple_unitary_vector"] = simple_unitary_vector

                            unitary_vector = [0 for i in range(6)]
                            combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                            if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                                unitary_vector[0] = 1
                            elif combo_unitary_types == ["federalism", "federalism"]:
                                unitary_vector[1] = 1
                            # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            #     unitary_vector[2] = 1
                            elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                                unitary_vector[3] = 1
                            elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                                unitary_vector[4] = 1
                            elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                                unitary_vector[5] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                            gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["2020_gdp"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                    gdp_idx_sign1 = 0
                                else:
                                    gdp_idx_sign1 = 1
                                if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                    gdp_idx_sign2 = 0
                                else:
                                    gdp_idx_sign2 = 1

                                if gdp_idx_sign1 == gdp_idx_sign2:
                                    gdp_converg = 1

                                    if gdp_idx_sign1 == 0:
                                        # take this class as reference
                                        # gdp_vec[0] = 1
                                        pass
                                    else:
                                        gdp_vec[1] = 1
                                else:
                                    gdp_converg = 0
                                    if gdp_idx_sign1 == 0:
                                        gdp_vec[2] = 1
                                    else:
                                        gdp_vec[3] = 1

                            else:
                                gdp_converg = float("nan")
                                gdp_vec = [float("nan") for i in range(4)]

                            # for test bug
                            if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                                print()

                            gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["gini_index"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["gini_index"][cur_idx1] < 35:
                                    gini_index_idx_sign1 = 0
                                else:
                                    gini_index_idx_sign1 = 1
                                if country_syn_dict["gini_index"][cur_idx2] < 35:
                                    gini_index_idx_sign2 = 0
                                else:
                                    gini_index_idx_sign2 = 1

                                if gini_index_idx_sign1 == gini_index_idx_sign2:
                                    gini_index_converg = 1

                                    if gini_index_idx_sign1 == 0:
                                        # take this class as reference
                                        # gini_index_vec[0] = 1
                                        pass
                                    else:
                                        gini_index_vec[1] = 1
                                else:
                                    gini_index_converg = 0

                                    if gini_index_idx_sign1 == 0:
                                        gini_index_vec[2] = 1
                                    else:
                                        gini_index_vec[3] = 1
                            else:
                                gini_index_converg = float("nan")
                                gini_index_vec = [float("nan") for i in range(4)]

                            dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["eiu"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["eiu"][cur_idx1] < 5:
                                    dem_idx_sign1 = 0
                                else:
                                    dem_idx_sign1 = 1
                                if country_syn_dict["eiu"][cur_idx2] < 5:
                                    dem_idx_sign2 = 0
                                else:
                                    dem_idx_sign2 = 1

                                if dem_idx_sign1 == dem_idx_sign2:
                                    dem_idx_converg = 1

                                    if dem_idx_sign1 == 0:
                                        # take this class as reference
                                        # dem_idx_vec[0] = 1
                                        pass
                                    else:
                                        dem_idx_vec[1] = 1
                                else:
                                    dem_idx_converg = 0

                                    if dem_idx_sign1 == 0:
                                        dem_idx_vec[2] = 1
                                    else:
                                        dem_idx_vec[3] = 1
                            else:
                                dem_idx_converg = float("nan")
                                dem_idx_vec = [float("nan") for i in range(4)]

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                            # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country1'] = country_alpha3_geography[cur_alpha3_1]["country_full_name"]
                            # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['country2'] = country_alpha3_geography[cur_alpha3_2]["country_full_name"]

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment"] = country_pair_investment_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

                model_sets = []
                for search_metric in ['vif','aic']:
                    inter_country_train_data = []
                    inter_country_train_label = []
                    fitlered_country_pair_num = 0
                    fitlered_sampling_pair_num = 0

                    country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                    for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                        for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                            cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                            cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                            cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                            cur_country_pair_sim = cur_country_pair["sim"]
                            cur_country_pair_same_lang = cur_country_pair["same_lang"]
                            cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                            cur_country_pair.pop('simple_lang_vec')
                            cur_country_pair.pop('lang_vec')
                            cur_country_pair.pop('sim')
                            cur_country_pair.pop('same_lang')
                            cur_country_pair.pop('same_spec_lang')


                            if np.isnan(cur_country_pair['border']):
                                continue
                            if np.isnan(cur_country_pair['diplomatic']):
                                continue
                            if np.isnan(cur_country_pair['investment']):
                                continue
                            if np.isnan(cur_country_pair['trade']):
                                continue
                            if np.isnan(cur_country_pair['immgration']):
                                continue

                            for attr in cur_country_pair.values():
                                if isinstance(attr, int) or isinstance(attr, float):
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                                # if isinstance(attr, list) and len(attr) == 1:
                                if isinstance(attr, list):
                                    country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                            # feature_combo_list = []
                            # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                            #     temp_list = []
                            #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                            #         temp_list.append(c)
                            #     feature_combo_list.extend(temp_list)

                            try:

                                cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] + cur_country_pair_simple_lang_vec + cur_country_pair_lang_vec
                                cur_train_data_row = cur_train_data_row[1:]
                                inter_country_train_data.append(cur_train_data_row)

                                inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                            except:
                                fitlered_country_pair_num += 1
                    print("fitlered_country_pair_num: ", fitlered_country_pair_num)

                    # normalization
                    minmax = MinMaxScaler()
                    inter_country_train_data = minmax.fit_transform(inter_country_train_data)

                    print("country pairs total number:", len(inter_country_train_label))

                    inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
                    inter_country_train_data_df.columns = ["Neighbor", "Geography distance","Continent sim", "Region1 sim",\
                                                        'Same political group','Both in NATO', 'Both in EU', 'Both in BRICS',\
                                                       'Across NATO-EU groups', 'Across NATO-BRICS groups', "Across EU-BRICS groups",\
                                                       "Same political system type", "Both republic", "Both federalism", \
                                                       "Neither republic nor federalism", "Republic and federalism", 'Republic and other', 'Federalism and other', \
                                                       "Same GDP category", "GDP(low-low)", "GDP(high-high)", "GDP(low->high)", "GDP(high->low)",\
                                                       "Same Gini Index category", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low->high)", "Gini Index(high->low)",\
                                                       "Same Democracy index category", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low->high)", "Democracy index(high->low)",\
                                                       "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                                       "Same language", "Speaking English", "Speaking German", "Speaking Spanish", "Speaking Polish", "Speaking French", \
                                                       "Speaking Chinese", "Speaking Arabic", "Speaking Turkish", "Speaking Italian", "Speaking Russian"]


                    # 0 is the pair number, 1 is the average similarity
                    # y = inter_country_train_data_df[1]
                    y = inter_country_train_label
                    X = inter_country_train_data_df

                    X_feature_names = X.columns.values.tolist()
                    valid_X_feature_names = []
                    # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                    for cur_feature in X_feature_names:
                        multivalue_sign = 0

                        cur_feature_values = X[cur_feature]
                        cmp_value = cur_feature_values[0]
                        for cur_feature_value in cur_feature_values:
                            if cur_feature_value != cmp_value:
                                multivalue_sign = 1
                                break
                        if multivalue_sign == 1:
                            valid_X_feature_names.append(cur_feature)

                    X = X[valid_X_feature_names]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

                    # Create an empty dictionary that will be used to store our results
                    function_dict = {'predictor': [], 'r-squared': []}
                    # Iterate through every column in X
                    cols = list(X.columns)
                    for col in cols:
                        # Create a dataframe called selected_X with only the 1 column
                        selected_X = X[[col]]
                        # Fit a model for our target and our selected column
                        model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                        # Predict what our target would be for our model
                        y_preds = model.predict(sm.add_constant(selected_X))
                        # Add the column name to our dictionary
                        function_dict['predictor'].append(col)
                        # Calculate the r-squared value between the target and predicted target
                        r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                        # Add the r-squared value to our dictionary
                        function_dict['r-squared'].append(r2)

                    # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
                    function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
                    # Display only the top 5 predictors

                    if search_metric == "vif":
                        selected_features = [function_df['predictor'].iat[0]]
                        features_to_ignore = []

                        # Since our function's ignore_features list is already empty, we don't need to
                        # include our features_to_ignore list.
                        while len(selected_features) + len(features_to_ignore) < len(cols):
                            next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                                     ignore_features=features_to_ignore)[0]
                            # check vif score
                            vif_factor = 5
                            temp_selected_features = selected_features + [next_feature]
                            temp_X = X[temp_selected_features]
                            temp_vif = pd.DataFrame()
                            temp_vif["features"] = temp_X.columns
                            temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                            cur_vif = temp_vif["VIF"].iat[-1]
                            if cur_vif <= vif_factor:
                                selected_features = temp_selected_features
                            else:
                                features_to_ignore.append(next_feature)
                    elif search_metric == "aic":
                        selected_features = [function_df['predictor'].iat[0]]
                        rest_features = []
                        for cur_feature in valid_X_feature_names:
                            if cur_feature not in selected_features:
                                rest_features.append(cur_feature)

                        best_aic = 10000000
                        search_max_time = 10000
                        search_time = 0
                        while len(selected_features) < len(cols) or search_time >= search_max_time:
                            # if there is no change in this turn then meaning no feature is selected.
                            # Should also stop search in this case
                            change_sign = 0
                            temp_feature_sets = [selected_features+[temp_feature] for temp_feature in rest_features]
                            for temp_feature_set in temp_feature_sets:
                                temp_X = X[temp_feature_set]
                                temp_model = sm.OLS(y, sm.add_constant(temp_X)).fit()
                                if temp_model.aic < best_aic:
                                    best_aic = temp_model.aic
                                    selected_features = temp_feature_set
                                    change_sign = 1
                            if change_sign == 0:
                                break
                            search_time += 1


                    model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                    model_sets.append(model)

                model_sets_res = summary_col(model_sets, stars=True)
                print("country:", select_country, "disaster:", cur_disaster_type)
                cur_save_folder = f"results/disaster_country_pair_per_country/{select_country}/{args.data}/"
                if not os.path.exists(cur_save_folder):
                    os.makedirs(cur_save_folder)

                with open(f"{cur_save_folder}/" + cur_disaster_type + '.txt', 'w') as f:
                    f.write(str(model_sets_res))
                print(model_sets_res.as_latex())


    if args.option == "rm_pair_summary":
        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass


        # investment flow, using absolute values in the summary
        country_pair_investment = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
        country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in
                                        country_pair_investment.keys()}
        country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
                country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
            except:
                pass


        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass



        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass


        cur_lang_set = [args.data[-2:]]
        for cur_lang in cur_lang_set:
            model_sets = []
            cur_lang_countries = []

            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            # only consider the models of countries which speak this language as official language


            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if (cur_alpha3_1 not in disaster_country_pair_stats_total) or (cur_alpha3_2 not in disaster_country_pair_stats_total[cur_alpha3_1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats_total[cur_alpha3_1][cur_alpha3_2]


                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                        # Region1 similarity
                        if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                        # language
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion') or (
                                country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(3)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                if gdp_idx_sign1 == 0:
                                    gdp_vec[2] = 1
                                else:
                                    gdp_vec[3] = 1

                            # confine dummy vector for the regression model of a certain country to have the reference class
                            # e.g. if the country is 'high', there will only be "low-high" and "high-high",
                            # treat "low-high" as the reference class,
                            # because for some country they don't have enough disaster or conflicts matching to "low" countries

                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_vec[0] = 0
                            else:
                                gdp_vec[2] = 0

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(4)]

                        # for test bug
                        if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                            print()

                        gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0

                                if gini_index_idx_sign1 == 0:
                                    gini_index_vec[2] = 1
                                else:
                                    gini_index_vec[3] = 1

                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_vec[0] = 0
                            else:
                                gini_index_vec[2] = 0
                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(4)]

                        dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0

                                if dem_idx_sign1 == 0:
                                    dem_idx_vec[2] = 1
                                else:
                                    dem_idx_vec[3] = 1

                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_vec[0] = 0
                            else:
                                dem_idx_vec[2] = 0
                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(4)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment"] = country_pair_investment_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            inter_country_train_data = []
            inter_country_train_label = []
            fitlered_country_pair_num = 0
            fitlered_sampling_pair_num = 0

            country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                    cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                    cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                    cur_country_pair_sim = cur_country_pair["sim"]
                    cur_country_pair_same_lang = cur_country_pair["same_lang"]
                    cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                    cur_country_pair.pop('simple_lang_vec')
                    cur_country_pair.pop('lang_vec')
                    cur_country_pair.pop('sim')
                    cur_country_pair.pop('same_lang')
                    cur_country_pair.pop('same_spec_lang')


                    if np.isnan(cur_country_pair['border']):
                        continue
                    if np.isnan(cur_country_pair['diplomatic']):
                        continue
                    if np.isnan(cur_country_pair['investment']):
                        continue
                    if np.isnan(cur_country_pair['trade']):
                        continue
                    if np.isnan(cur_country_pair['immgration']):
                        continue

                    for attr in cur_country_pair.values():
                        if isinstance(attr, int) or isinstance(attr, float):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                        # if isinstance(attr, list) and len(attr) == 1:
                        if isinstance(attr, list):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                    # feature_combo_list = []
                    # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                    #     temp_list = []
                    #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                    #         temp_list.append(c)
                    #     feature_combo_list.extend(temp_list)

                    try:

                        cur_train_data_row = [cur_country_pair['border']] + [cur_country_pair['continent_sim']] + cur_country_pair['gdp_vec'] + cur_country_pair['gini_index_vec'] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['diplomatic']] + [cur_country_pair['investment']] + [cur_country_pair['trade']] + [cur_country_pair['immgration']] + cur_country_pair_simple_lang_vec
                        inter_country_train_data.append(cur_train_data_row)

                        inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                    except:
                        traceback.print_exc()
                        fitlered_country_pair_num += 1
            print("fitlered_country_pair_num: ", fitlered_country_pair_num)


            # normalization
            minmax = MinMaxScaler()

            # using absolute value to simplify the trading effect
            inter_country_train_data = np.abs(inter_country_train_data)
            inter_country_train_data = list(inter_country_train_data)

            inter_country_train_data = minmax.fit_transform(inter_country_train_data)


            print("country pairs total number:", len(inter_country_train_label))

            inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
            inter_country_train_data_df.columns = ["Neighbor", "Continent sim", \
                                               "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                               "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                               "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                               "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                               "Same language"]



            # 0 is the pair number, 1 is the average similarity
            # y = inter_country_train_data_df[1]
            y = inter_country_train_label
            X = inter_country_train_data_df

            X_feature_names = X.columns.values.tolist()
            valid_X_feature_names = []
            # filter the columns with only 1 kind of value, e.g. all 0 or all 1
            for cur_feature in X_feature_names:
                multivalue_sign = 0

                cur_feature_values = X[cur_feature]
                cmp_value = cur_feature_values[0]
                for cur_feature_value in cur_feature_values:
                    if cur_feature_value != cmp_value:
                        multivalue_sign = 1
                        break
                if multivalue_sign == 1:
                    valid_X_feature_names.append(cur_feature)

            X = X[valid_X_feature_names]
            print("Shape of X:", X.shape)
            model = sm.OLS(y, sm.add_constant(X)).fit()
            model_sets.append(model)

            model_sets_res = summary_col(model_sets, model_names = cur_lang_countries,stars=True, float_format='%0.4f', info_dict={'Observation Num': nobs})
            cur_save_folder = f"results/disaster_lang_summary/{cur_lang}/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_lang + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())


    if args.option == "rm_pair_lang_summary":

        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass


        # investment flow, using absolute values in the summary
        country_pair_investment_2018 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment_2018 = country_pair_investment_2018[country_pair_investment_2018["INDICATOR"] == "VALUE"]
        country_pair_investment_2018 = country_pair_investment_2018[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_2018_dict = {key: country_pair_investment_2018[key].to_list() for key in
                                        country_pair_investment_2018.keys()}
        country_pair_investment_2018_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_2018_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["PARTNER_CTRY"][country_idx]]

                # country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_2018_dict["2018"][country_idx]
                country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2018_dict["2018"][country_idx])
                # country_pair_investment_2018_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                # abs(country_pair_investment_2018_dict["2018"][country_idx])
            except:
                pass

        # # investment of 2005, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2005 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2005 = country_pair_investment_2005[country_pair_investment_2005["INDICATOR"] == "VALUE"]
        # country_pair_investment_2005 = country_pair_investment_2005[["REPORT_CTRY", "PARTNER_CTRY", "2005"]]
        #
        # country_pair_investment_2005_dict = {key: country_pair_investment_2005[key].to_list() for key in country_pair_investment_2005.keys()}
        # country_pair_investment_2005_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2005_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2005_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #         country_pair_investment_2005_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #     except:
        #         pass

        # # investment of 2008, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2008 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2008 = country_pair_investment_2008[country_pair_investment_2008["INDICATOR"] == "VALUE"]
        # country_pair_investment_2008 = country_pair_investment_2008[["REPORT_CTRY", "PARTNER_CTRY", "2008"]]
        #
        # country_pair_investment_2008_dict = {key: country_pair_investment_2008[key].to_list() for key in country_pair_investment_2008.keys()}
        # country_pair_investment_2008_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2008_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2008_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #         country_pair_investment_2008_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #     except:
        #         pass


        '''investment flow from the OECD, using absolute values in the summary'''
        # country_pair_investment_oecd = pd.read_csv("data/2018-2022-fdi-outflows-clean.csv")
        # country_pair_investment_oecd = country_pair_investment_oecd[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]
        #
        # country_pair_investment_oecd_dict = {key: country_pair_investment_oecd[key].to_list() for key in
        #                                 country_pair_investment_oecd.keys()}
        # country_pair_investment_oecd_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_oecd_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = country_pair_investment_oecd_dict["REPORT_CTRY"][country_idx]
        #         cur_sending_alpha3 = country_pair_investment_oecd_dict["PARTNER_CTRY"][country_idx]
        #
        #         country_pair_investment_oecd_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #         # country_pair_investment_oecd_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #     except:
        #         pass

        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass



        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass


        cur_lang_set = [args.data[-2:]]
        for cur_lang in cur_lang_set:
            model_sets = []
            cur_lang_countries = []
            valid_pair_count_dict = {}

            report_country_count_dict = defaultdict(int) # compute the number of countries where the disasters/conflicts covered by each reporting country

            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

            # # only consider the models of countries which speak this language as official language
            # if cur_lang not in top_country_list[select_country]:
            #     continue


            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if not (disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2] >= 0):
                            continue

                        if (cur_alpha3_1 not in disaster_country_pair_stats_total) or (cur_alpha3_2 not in disaster_country_pair_stats_total[cur_alpha3_1]):
                            # print()
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats_total[cur_alpha3_1][cur_alpha3_2]
                            report_country_count_dict[cur_alpha3_2] += 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["death_num"] = disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2]

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                        # Region1 similarity
                        if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                        # language
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion') or (
                                country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(3)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                if gdp_idx_sign1 == 0:
                                    gdp_vec[2] = 1
                                else:
                                    gdp_vec[3] = 1

                            # confine dummy vector for the regression model of a certain country to have the reference class
                            # e.g. if the country is 'high', there will only be "low-high" and "high-high",
                            # treat "low-high" as the reference class,
                            # because for some country they don't have enough disaster or conflicts matching to "low" countries

                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_vec[0] = 0
                            else:
                                gdp_vec[2] = 0

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(4)]

                        # for test bug
                        if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                            print()

                        gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0

                                if gini_index_idx_sign1 == 0:
                                    gini_index_vec[2] = 1
                                else:
                                    gini_index_vec[3] = 1

                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_vec[0] = 0
                            else:
                                gini_index_vec[2] = 0
                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(4)]

                        dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0

                                if dem_idx_sign1 == 0:
                                    dem_idx_vec[2] = 1
                                else:
                                    dem_idx_vec[3] = 1

                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_vec[0] = 0
                            else:
                                dem_idx_vec[2] = 0
                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(4)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_2018_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2005"] = country_pair_investment_2005_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2008"] = country_pair_investment_2008_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_oecd_value_dict[cur_alpha3_1][cur_alpha3_2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            valid_country_dict = {}
            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if report_country_count_dict[cur_alpha3_2] < valid_pair_bound or (cur_alpha3_2 not in top_country_list) or (cur_alpha3_2 in top_country_list and cur_lang not in top_country_list[cur_alpha3_2]):
                            if cur_alpha3_1 in country_pair_syn_dict_dict_dict and cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                                del country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]
                        else:
                            valid_country_dict[cur_alpha3_2] = report_country_count_dict[cur_alpha3_2]

            print("country coverage number list:")
            for cur_alpha3_2 in valid_country_dict:
                print(cur_alpha3_2, "  ", valid_country_dict[cur_alpha3_2])

            inter_country_train_data = []
            inter_country_train_label = []
            fitlered_country_pair_num = 0
            fitlered_sampling_pair_num = 0

            country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                    cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                    cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                    cur_country_pair_sim = cur_country_pair["sim"]
                    cur_country_pair_same_lang = cur_country_pair["same_lang"]
                    cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                    cur_country_pair.pop('simple_lang_vec')
                    cur_country_pair.pop('lang_vec')
                    cur_country_pair.pop('sim')
                    cur_country_pair.pop('same_lang')
                    cur_country_pair.pop('same_spec_lang')


                    if np.isnan(cur_country_pair['border']):
                        continue
                    if np.isnan(cur_country_pair['diplomatic']):
                        continue
                    if np.isnan(cur_country_pair['investment_2018']):
                        continue
                    # if np.isnan(cur_country_pair['investment_2005']):
                    #     continue
                    # if np.isnan(cur_country_pair['investment_2008']):
                    #     continue
                    if np.isnan(cur_country_pair['trade']):
                        continue
                    if np.isnan(cur_country_pair['immgration']):
                        continue

                    for attr in cur_country_pair.values():
                        if isinstance(attr, int) or isinstance(attr, float):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                        # if isinstance(attr, list) and len(attr) == 1:
                        if isinstance(attr, list):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                    # feature_combo_list = []
                    # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                    #     temp_list = []
                    #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                    #         temp_list.append(c)
                    #     feature_combo_list.extend(temp_list)

                    try:
                        cur_train_data_row = [cur_country_pair['death_num']] + [cur_country_pair['border']] + [cur_country_pair['continent_sim']] + cur_country_pair['gdp_vec'] + cur_country_pair['gini_index_vec'] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['diplomatic']] + [cur_country_pair['investment_2018']] + [cur_country_pair['trade']] + [cur_country_pair['immgration']] + cur_country_pair_simple_lang_vec
                        inter_country_train_data.append(cur_train_data_row)

                        inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                    except:
                        traceback.print_exc()
                        fitlered_country_pair_num += 1
            print("fitlered_country_pair_num: ", fitlered_country_pair_num)


            # normalization
            minmax = MinMaxScaler()

            # using absolute value to simplify the trading effect
            inter_country_train_data = np.abs(inter_country_train_data)
            inter_country_train_data = list(inter_country_train_data)

            inter_country_train_data = minmax.fit_transform(inter_country_train_data)


            print("country pairs total number:", len(inter_country_train_label))

            inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
            inter_country_train_data_df.columns = ["Death num", "Neighbor", "Continent sim", \
                                               "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                               "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                               "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                               "Diplomatic Relation", "Investment_2018", "Trade", "Immigration", \
                                               "Same language"]



            # 0 is the pair number, 1 is the average similarity
            # y = inter_country_train_data_df[1]
            y = inter_country_train_label
            X = inter_country_train_data_df

            X_feature_names = X.columns.values.tolist()
            valid_X_feature_names = []
            # filter the columns with only 1 kind of value, e.g. all 0 or all 1
            for cur_feature in X_feature_names:
                multivalue_sign = 0

                cur_feature_values = X[cur_feature]
                cmp_value = cur_feature_values[0]
                for cur_feature_value in cur_feature_values:
                    if cur_feature_value != cmp_value:
                        multivalue_sign = 1
                        break
                if multivalue_sign == 1:
                    valid_X_feature_names.append(cur_feature)

            X = X[valid_X_feature_names]
            print("Shape of X:", X.shape)
            model = sm.OLS(y, sm.add_constant(X)).fit()
            model_sets.append(model)

            model_sets_res = summary_col(model_sets, model_names = cur_lang_countries,stars=True, float_format='%0.4f', info_dict={'Observation Num': nobs})
            cur_save_folder = f"results/disaster_lang_select_summary/{cur_lang}/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_lang + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())


            print("positive samples of each country:")
            print(report_country_count_dict)

    if args.option == "rm_pair_mix_factor_lang_summary":

        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass


        # investment flow, using absolute values in the summary
        country_pair_investment_2018 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment_2018 = country_pair_investment_2018[country_pair_investment_2018["INDICATOR"] == "VALUE"]
        country_pair_investment_2018 = country_pair_investment_2018[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_2018_dict = {key: country_pair_investment_2018[key].to_list() for key in
                                        country_pair_investment_2018.keys()}
        country_pair_investment_2018_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_2018_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["PARTNER_CTRY"][country_idx]]

                # country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_2018_dict["2018"][country_idx]
                country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2018_dict["2018"][country_idx])
                # country_pair_investment_2018_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                # abs(country_pair_investment_2018_dict["2018"][country_idx])
            except:
                pass

        # # investment of 2005, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2005 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2005 = country_pair_investment_2005[country_pair_investment_2005["INDICATOR"] == "VALUE"]
        # country_pair_investment_2005 = country_pair_investment_2005[["REPORT_CTRY", "PARTNER_CTRY", "2005"]]
        #
        # country_pair_investment_2005_dict = {key: country_pair_investment_2005[key].to_list() for key in country_pair_investment_2005.keys()}
        # country_pair_investment_2005_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2005_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2005_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #         country_pair_investment_2005_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #     except:
        #         pass

        # # investment of 2008, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2008 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2008 = country_pair_investment_2008[country_pair_investment_2008["INDICATOR"] == "VALUE"]
        # country_pair_investment_2008 = country_pair_investment_2008[["REPORT_CTRY", "PARTNER_CTRY", "2008"]]
        #
        # country_pair_investment_2008_dict = {key: country_pair_investment_2008[key].to_list() for key in country_pair_investment_2008.keys()}
        # country_pair_investment_2008_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2008_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2008_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #         country_pair_investment_2008_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #     except:
        #         pass


        '''investment flow from the OECD, using absolute values in the summary'''
        # country_pair_investment_oecd = pd.read_csv("data/2018-2022-fdi-outflows-clean.csv")
        # country_pair_investment_oecd = country_pair_investment_oecd[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]
        #
        # country_pair_investment_oecd_dict = {key: country_pair_investment_oecd[key].to_list() for key in
        #                                 country_pair_investment_oecd.keys()}
        # country_pair_investment_oecd_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_oecd_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = country_pair_investment_oecd_dict["REPORT_CTRY"][country_idx]
        #         cur_sending_alpha3 = country_pair_investment_oecd_dict["PARTNER_CTRY"][country_idx]
        #
        #         country_pair_investment_oecd_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #         # country_pair_investment_oecd_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #     except:
        #         pass

        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass



        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass


        cur_lang_set = [args.data[-2:]]
        for cur_lang in cur_lang_set:
            model_sets = []
            cur_lang_countries = []
            valid_pair_count_dict = {}

            report_country_count_dict = defaultdict(int) # compute the number of countries where the disasters/conflicts covered by each reporting country

            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            country_syn_dict_dict_dict = defaultdict(int)
            # # only consider the models of countries which speak this language as official language
            # if cur_lang not in top_country_list[select_country]:
            #     continue

            for event_class in disaster_country_stats:
                print(event_class)

            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if not (disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2] >= 0):
                            continue

                        # if not (cur_alpha3_2 == "USA"):
                        #     continue
                        if (cur_alpha3_1 not in disaster_country_pair_stats_total) or (cur_alpha3_2 not in disaster_country_pair_stats_total[cur_alpha3_1]):
                            # print()
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats_total[cur_alpha3_1][cur_alpha3_2]
                            report_country_count_dict[cur_alpha3_2] += 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["death_num"] = disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2]

                        event_class_list_vec = []
                        event_class_dist_vec = []
                        for event_class in disaster_country_stats:
                            event_class_list_vec.append(disaster_country_event_class_dict[cur_alpha3_1][cur_alpha3_2][event_class])
                        for event_class_count in event_class_list_vec:
                            if sum(event_class_list_vec) > 0:
                                event_class_dist_vec.append(event_class_count/sum(event_class_list_vec))
                            else:
                                event_class_dist_vec.append(0)
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['event_class_vec'] = event_class_dist_vec

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        # geographic distance
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent one-hot vector
                        continent_vec = [0 for i in range(len(continent_list))]
                        if country_syn_dict["Continent"][cur_idx1] in continent_list:
                            # Find the index of the continent in the list
                            index = continent_list.index(country_syn_dict["Continent"][cur_idx1])
                            # Set the corresponding position in the vector to 1
                            continent_vec[index] = 1

                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0


                        # Region1 similarity
                        if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                        # language
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion') or (
                                country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(3)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        # gdp binary
                        if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                            gdp_binary = 0
                        else:
                            gdp_binary = 1

                        # gdp pair-wise vector
                        gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                if gdp_idx_sign1 == 0:
                                    gdp_vec[2] = 1
                                else:
                                    gdp_vec[3] = 1

                            # confine dummy vector for the regression model of a certain country to have the reference class
                            # e.g. if the country is 'high', there will only be "low-high" and "high-high",
                            # treat "low-high" as the reference class,
                            # because for some country they don't have enough disaster or conflicts matching to "low" countries

                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_vec[0] = 0
                            else:
                                gdp_vec[2] = 0

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(4)]

                        # for test bug
                        if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                            print()

                        # gini binary
                        if country_syn_dict["gini_index"][cur_idx1] < 35:
                            gini_binary = 0
                        else:
                            gini_binary = 1

                        gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0

                                if gini_index_idx_sign1 == 0:
                                    gini_index_vec[2] = 1
                                else:
                                    gini_index_vec[3] = 1

                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_vec[0] = 0
                            else:
                                gini_index_vec[2] = 0
                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(4)]

                        # dem binary
                        if country_syn_dict["eiu"][cur_idx1] < 5:
                            dem_binary = 0
                        else:
                            dem_binary = 1

                        dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0

                                if dem_idx_sign1 == 0:
                                    dem_idx_vec[2] = 1
                                else:
                                    dem_idx_vec[3] = 1

                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_vec[0] = 0
                            else:
                                dem_idx_vec[2] = 0
                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(4)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["continent_vec"] = continent_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_binary"] = gdp_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_binary"] = gini_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_binary"] = dem_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_2018_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2005"] = country_pair_investment_2005_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2008"] = country_pair_investment_2008_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_oecd_value_dict[cur_alpha3_1][cur_alpha3_2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            valid_country_dict = {}
            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if report_country_count_dict[cur_alpha3_2] < valid_pair_bound or (cur_alpha3_2 not in top_country_list) or (cur_alpha3_2 in top_country_list and cur_lang not in top_country_list[cur_alpha3_2]):
                        # if report_country_count_dict[cur_alpha3_2] < valid_pair_bound:
                            if cur_alpha3_1 in country_pair_syn_dict_dict_dict and cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                                del country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]
                        else:
                            valid_country_dict[cur_alpha3_2] = report_country_count_dict[cur_alpha3_2]

            print("country coverage number list:")
            for cur_alpha3_2 in valid_country_dict:
                print(cur_alpha3_2, "  ", valid_country_dict[cur_alpha3_2])

            inter_country_train_data = []
            inter_country_train_data_to_save = []
            inter_country_train_label = []
            fitlered_country_pair_num = 0
            fitlered_sampling_pair_num = 0

            country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                    cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                    cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                    cur_country_pair_sim = cur_country_pair["sim"]
                    cur_country_pair_same_lang = cur_country_pair["same_lang"]
                    cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                    cur_country_pair.pop('simple_lang_vec')
                    cur_country_pair.pop('lang_vec')
                    cur_country_pair.pop('sim')
                    cur_country_pair.pop('same_lang')
                    cur_country_pair.pop('same_spec_lang')


                    if np.isnan(cur_country_pair['border']):
                        continue
                    if np.isnan(cur_country_pair['diplomatic']):
                        continue
                    if np.isnan(cur_country_pair['investment_2018']):
                        continue
                    # if np.isnan(cur_country_pair['investment_2005']):
                    #     continue
                    # if np.isnan(cur_country_pair['investment_2008']):
                    #     continue
                    if np.isnan(cur_country_pair['trade']):
                        continue
                    if np.isnan(cur_country_pair['immgration']):
                        continue

                    for attr in cur_country_pair.values():
                        if isinstance(attr, int) or isinstance(attr, float):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                        # if isinstance(attr, list) and len(attr) == 1:
                        if isinstance(attr, list):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                    # feature_combo_list = []
                    # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                    #     temp_list = []
                    #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                    #         temp_list.append(c)
                    #     feature_combo_list.extend(temp_list)

                    try:
                        cur_train_data_row = [cur_country_pair['death_num']] + [cur_country_pair['border']] + cur_country_pair['continent_vec'] + [cur_country_pair['continent_sim']] + [cur_country_pair['gdp_binary']] + cur_country_pair['gdp_vec'] + [cur_country_pair['gini_binary']] + cur_country_pair['gini_index_vec'] + [cur_country_pair['dem_binary']] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['diplomatic']] + [cur_country_pair['investment_2018']] + [cur_country_pair['trade']] + [cur_country_pair['immgration']] + cur_country_pair_simple_lang_vec
                        inter_country_train_data.append(cur_train_data_row)
                        inter_country_train_data_to_save.append([cur_alpha3_1, cur_alpha3_2] + cur_train_data_row)

                        inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                    except:
                        traceback.print_exc()
                        fitlered_country_pair_num += 1
            print("fitlered_country_pair_num: ", fitlered_country_pair_num)


            # normalization
            minmax = MinMaxScaler()

            # using absolute value to simplify the trading effect
            inter_country_train_data = np.abs(inter_country_train_data)
            inter_country_train_data = list(inter_country_train_data)

            inter_country_train_data = minmax.fit_transform(inter_country_train_data)


            print("country pairs total number:", len(inter_country_train_label))

            inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
            inter_country_train_data_df.columns = ["Death num",  \
                    "Neighbor", "Asia", "Europe", "Africa", "Oceania", "North America", "South America", "Antarctica", "Continent sim", \
                                               "GDP(high)", "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                               "Gini Index(high)", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                               "Democracy index(high)", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                               "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                               "Same language"]
            # inter_country_train_data_df.columns = ["Death num", "storm", "flood", "landslide", "earthquake", "avalanche", "wildfire", "volcano", \
            #         "Neighbor", "Asia", "Europe", "Africa", "Oceania", "North America", "South America", "Antarctica", "Continent sim", \
            #                                    "GDP(high)", "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
            #                                    "Gini Index(high)", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
            #                                    "Democracy index(high)", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
            #                                    "Diplomatic Relation", "Investment", "Trade", "Immigration", \
            #                                    "Same language"]x

            inter_country_train_data_to_save_df = pd.DataFrame(inter_country_train_data_to_save)
            inter_country_train_data_to_save_df.columns = ["event country", "reporting country", "Death num",  \
                    "Neighbor", "Asia", "Europe", "Africa", "Oceania", "North America", "South America", "Antarctica", "Continent sim", \
                                               "GDP(high)", "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                               "Gini Index(high)", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                               "Democracy index(high)", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                               "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                               "Same language"]
            inter_country_train_data_to_save_df.to_csv(release_route)

            # 0 is the pair number, 1 is the average similarity
            # y = inter_country_train_data_df[1]
            y = inter_country_train_label
            X = inter_country_train_data_df

            X_feature_names = X.columns.values.tolist()
            valid_X_feature_names = []
            # filter the columns with only 1 kind of value, e.g. all 0 or all 1
            for cur_feature in X_feature_names:
                multivalue_sign = 0

                cur_feature_values = X[cur_feature]
                cmp_value = cur_feature_values[0]
                for cur_feature_value in cur_feature_values:
                    if cur_feature_value != cmp_value:
                        multivalue_sign = 1
                        break
                if multivalue_sign == 1:
                    valid_X_feature_names.append(cur_feature)

            X = X[valid_X_feature_names]
            print("Shape of X:", X.shape)

            # Create an empty dictionary that will be used to store our results
            function_dict = {'predictor': [], 'r-squared': []}
            # Iterate through every column in X
            cols = list(X.columns)
            for col in cols:
                # Create a dataframe called selected_X with only the 1 column
                selected_X = X[[col]]
                # Fit a model for our target and our selected column
                model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                # Predict what our target would be for our model
                y_preds = model.predict(sm.add_constant(selected_X))
                # Add the column name to our dictionary
                function_dict['predictor'].append(col)
                # Calculate the r-squared value between the target and predicted target
                r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                # Add the r-squared value to our dictionary
                function_dict['r-squared'].append(r2)

            # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
            function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
            # Display only the top 5 predictors

            selected_features = [function_df['predictor'].iat[0]]
            features_to_ignore = []

            # Since our function's ignore_features list is already empty, we don't need to
            # include our features_to_ignore list.
            while len(selected_features) + len(features_to_ignore) < len(cols):
                next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                     ignore_features=features_to_ignore)[0]
                # check vif score
                vif_factor = 5
                temp_selected_features = selected_features + [next_feature]
                temp_X = X[temp_selected_features]
                temp_vif = pd.DataFrame()
                temp_vif["features"] = temp_X.columns
                temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                cur_vif = temp_vif["VIF"].iat[-1]
                if cur_vif <= vif_factor:
                    selected_features = temp_selected_features
                else:
                    features_to_ignore.append(next_feature)

            # model = sm.OLS(y, sm.add_constant(X)).fit()
            model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
            model_sets.append(model)

            model_sets_res = summary_col(model_sets, model_names = cur_lang_countries,stars=True, float_format='%0.4f', info_dict={'Observation Num': nobs})
            cur_save_folder = f"results/disaster_lang_select_summary/{cur_lang}/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_lang + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())


            print("positive samples of each country:")
            print(report_country_count_dict)

    if args.option == "rm_pair_mix_factor_us_summary":
        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass


        # investment flow, using absolute values in the summary
        country_pair_investment_2018 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment_2018 = country_pair_investment_2018[country_pair_investment_2018["INDICATOR"] == "VALUE"]
        country_pair_investment_2018 = country_pair_investment_2018[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_2018_dict = {key: country_pair_investment_2018[key].to_list() for key in
                                        country_pair_investment_2018.keys()}
        country_pair_investment_2018_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_2018_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2018_dict["PARTNER_CTRY"][country_idx]]

                # country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_investment_2018_dict["2018"][country_idx]
                country_pair_investment_2018_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2018_dict["2018"][country_idx])
                # country_pair_investment_2018_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                # abs(country_pair_investment_2018_dict["2018"][country_idx])
            except:
                pass

        # # investment of 2005, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2005 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2005 = country_pair_investment_2005[country_pair_investment_2005["INDICATOR"] == "VALUE"]
        # country_pair_investment_2005 = country_pair_investment_2005[["REPORT_CTRY", "PARTNER_CTRY", "2005"]]
        #
        # country_pair_investment_2005_dict = {key: country_pair_investment_2005[key].to_list() for key in country_pair_investment_2005.keys()}
        # country_pair_investment_2005_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2005_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2005_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2005_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #         country_pair_investment_2005_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2005_dict["2005"][country_idx])
        #     except:
        #         pass

        # # investment of 2008, as comparison to investment flow to indicate the reversed causality
        # country_pair_investment_2008 = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        # country_pair_investment_2008 = country_pair_investment_2008[country_pair_investment_2008["INDICATOR"] == "VALUE"]
        # country_pair_investment_2008 = country_pair_investment_2008[["REPORT_CTRY", "PARTNER_CTRY", "2008"]]
        #
        # country_pair_investment_2008_dict = {key: country_pair_investment_2008[key].to_list() for key in country_pair_investment_2008.keys()}
        # country_pair_investment_2008_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_2008_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["REPORT_CTRY"][country_idx]]
        #         cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_2008_dict["PARTNER_CTRY"][country_idx]]
        #
        #         country_pair_investment_2008_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #         country_pair_investment_2008_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_2008_dict["2008"][country_idx])
        #     except:
        #         pass


        '''investment flow from the OECD, using absolute values in the summary'''
        # country_pair_investment_oecd = pd.read_csv("data/2018-2022-fdi-outflows-clean.csv")
        # country_pair_investment_oecd = country_pair_investment_oecd[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]
        #
        # country_pair_investment_oecd_dict = {key: country_pair_investment_oecd[key].to_list() for key in
        #                                 country_pair_investment_oecd.keys()}
        # country_pair_investment_oecd_value_dict = defaultdict(lambda: defaultdict(float))
        # for country_idx in range(len(country_pair_investment_oecd_dict["REPORT_CTRY"])):
        #     try:
        #         cur_destination_alpha3 = country_pair_investment_oecd_dict["REPORT_CTRY"][country_idx]
        #         cur_sending_alpha3 = country_pair_investment_oecd_dict["PARTNER_CTRY"][country_idx]
        #
        #         country_pair_investment_oecd_value_dict[cur_destination_alpha3][cur_sending_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #         # country_pair_investment_oecd_value_dict[cur_sending_alpha3][cur_destination_alpha3] += abs(country_pair_investment_oecd_dict["2018"][country_idx])
        #     except:
        #         pass

        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass



        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass


        cur_lang_set = [args.data[-2:]]
        for cur_lang in cur_lang_set:
            model_sets = []
            cur_lang_countries = []
            valid_pair_count_dict = {}

            report_country_count_dict = defaultdict(int) # compute the number of countries where the disasters/conflicts covered by each reporting country

            country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            country_syn_dict_dict_dict = defaultdict(int)
            # # only consider the models of countries which speak this language as official language
            # if cur_lang not in top_country_list[select_country]:
            #     continue


            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if not (disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2] >= 0):
                            continue

                        if (cur_alpha3_1 not in disaster_country_pair_stats_total) or (cur_alpha3_2 not in disaster_country_pair_stats_total[cur_alpha3_1]):
                            # print()
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats_total[cur_alpha3_1][cur_alpha3_2]
                            report_country_count_dict[cur_alpha3_2] += 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["death_num"] = disaster_country_pair_stats_death_total[cur_alpha3_1][cur_alpha3_2]

                        if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                        # geographic distance
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                        # continent one-hot vector
                        continent_vec = [0 for i in range(len(continent_list))]
                        if country_syn_dict["Continent"][cur_idx1] in continent_list:
                            # Find the index of the continent in the list
                            index = continent_list.index(country_syn_dict["Continent"][cur_idx1])
                            # Set the corresponding position in the vector to 1
                            continent_vec[index] = 1


                        # continent similarity
                        if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0


                        # Region1 similarity
                        if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                        else:
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                        # language
                        try:
                            common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                        except:
                            common_official_langs_family = []

                        simple_lang_vector = [0 for i in range(1)]
                        if common_official_langs != []:
                            simple_lang_vector[0] = 1
                        # elif common_official_langs_family != []:
                        #     simple_lang_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                        lang_vector = [0 for i in range(10)]
                        if common_official_langs != []:
                            lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                            for lang in common_official_langs:
                                if lang in lang_full_name_map_keys:
                                    lang_vector[lang_full_name_map_keys.index(lang)] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics', 'five_eyes']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        across_group = []
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion') or (
                                country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-eunion')
                        if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                            cur_idx2] == 'nato'):
                            across_group.append('nato-brics')
                        if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                            cur_idx2] == 'brics') or (
                                country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                            cur_idx2] == 'eunion'):
                            across_group.append('eunion-brics')

                        simple_political_group_vector = [0]
                        if common_group != []:
                            simple_political_group_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                        '''political group featrues'''
                        political_group_vector = [0 for i in range(4)]
                        if 'nato' in common_group:
                            political_group_vector[0] = 1
                        elif 'eunion' in common_group:
                            political_group_vector[1] = 1
                        elif 'brics' in common_group:
                            political_group_vector[2] = 1
                        elif 'five_eyes' in common_group:
                            political_group_vector[3] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["political_group_vec"] = political_group_vector

                        across_group_vector = [0 for i in range(3)]
                        if 'nato-eunion' in across_group:
                            across_group_vector[0] = 1
                        if 'nato-brics' in across_group:
                            across_group_vector[1] = 1
                        if 'eunion-brics' in across_group:
                            across_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                        # unitary type combination categories, 6 kinds
                        simple_unitary_vector = [0]
                        if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                            simple_unitary_vector[0] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "simple_unitary_vector"] = simple_unitary_vector

                        unitary_vector = [0 for i in range(6)]
                        combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                        if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                            unitary_vector[0] = 1
                        elif combo_unitary_types == ["federalism", "federalism"]:
                            unitary_vector[1] = 1
                        # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                        #     unitary_vector[2] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[3] = 1
                        elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            unitary_vector[4] = 1
                        elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                            unitary_vector[5] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        # gdp binary
                        if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                            gdp_binary = 0
                        else:
                            gdp_binary = 1

                        # gdp pair-wise vector
                        gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                country_syn_dict["2020_gdp"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                gdp_idx_sign1 = 0
                            else:
                                gdp_idx_sign1 = 1
                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_idx_sign2 = 0
                            else:
                                gdp_idx_sign2 = 1

                            if gdp_idx_sign1 == gdp_idx_sign2:
                                gdp_converg = 1

                                if gdp_idx_sign1 == 0:
                                    # take this class as reference
                                    # gdp_vec[0] = 1
                                    pass
                                else:
                                    gdp_vec[1] = 1
                            else:
                                gdp_converg = 0
                                if gdp_idx_sign1 == 0:
                                    gdp_vec[2] = 1
                                else:
                                    gdp_vec[3] = 1

                            # confine dummy vector for the regression model of a certain country to have the reference class
                            # e.g. if the country is 'high', there will only be "low-high" and "high-high",
                            # treat "low-high" as the reference class,
                            # because for some country they don't have enough disaster or conflicts matching to "low" countries

                            if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                gdp_vec[0] = 0
                            else:
                                gdp_vec[2] = 0

                        else:
                            gdp_converg = float("nan")
                            gdp_vec = [float("nan") for i in range(4)]

                        # for test bug
                        if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                            print()

                        # gini binary
                        if country_syn_dict["gini_index"][cur_idx1] < 35:
                            gini_binary = 0
                        else:
                            gini_binary = 1

                        gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                country_syn_dict["gini_index"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["gini_index"][cur_idx1] < 35:
                                gini_index_idx_sign1 = 0
                            else:
                                gini_index_idx_sign1 = 1
                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_idx_sign2 = 0
                            else:
                                gini_index_idx_sign2 = 1

                            if gini_index_idx_sign1 == gini_index_idx_sign2:
                                gini_index_converg = 1

                                if gini_index_idx_sign1 == 0:
                                    # take this class as reference
                                    # gini_index_vec[0] = 1
                                    pass
                                else:
                                    gini_index_vec[1] = 1
                            else:
                                gini_index_converg = 0

                                if gini_index_idx_sign1 == 0:
                                    gini_index_vec[2] = 1
                                else:
                                    gini_index_vec[3] = 1

                            if country_syn_dict["gini_index"][cur_idx2] < 35:
                                gini_index_vec[0] = 0
                            else:
                                gini_index_vec[2] = 0
                        else:
                            gini_index_converg = float("nan")
                            gini_index_vec = [float("nan") for i in range(4)]

                        # dem binary
                        if country_syn_dict["eiu"][cur_idx1] < 5:
                            dem_binary = 0
                        else:
                            dem_binary = 1

                        dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                        if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                country_syn_dict["eiu"][cur_idx2]):
                            # 0 stands for low, 1 stands for high
                            if country_syn_dict["eiu"][cur_idx1] < 5:
                                dem_idx_sign1 = 0
                            else:
                                dem_idx_sign1 = 1
                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_sign2 = 0
                            else:
                                dem_idx_sign2 = 1

                            if dem_idx_sign1 == dem_idx_sign2:
                                dem_idx_converg = 1

                                if dem_idx_sign1 == 0:
                                    # take this class as reference
                                    # dem_idx_vec[0] = 1
                                    pass
                                else:
                                    dem_idx_vec[1] = 1
                            else:
                                dem_idx_converg = 0

                                if dem_idx_sign1 == 0:
                                    dem_idx_vec[2] = 1
                                else:
                                    dem_idx_vec[3] = 1

                            if country_syn_dict["eiu"][cur_idx2] < 5:
                                dem_idx_vec[0] = 0
                            else:
                                dem_idx_vec[2] = 0
                        else:
                            dem_idx_converg = float("nan")
                            dem_idx_vec = [float("nan") for i in range(4)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["continent_vec"] = continent_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_binary"] = gdp_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_binary"] = gini_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_binary"] = dem_binary
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_2018_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2005"] = country_pair_investment_2005_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2008"] = country_pair_investment_2008_value_dict[cur_alpha3_1][cur_alpha3_2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment_2018"] = country_pair_investment_oecd_value_dict[cur_alpha3_1][cur_alpha3_2]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

            valid_country_dict = {}
            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        continue
                    else:
                        cur_idx1 = i
                        cur_idx2 = j

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        if report_country_count_dict[cur_alpha3_2] < valid_pair_bound or (cur_alpha3_2 not in us_country_list) or (cur_alpha3_2 in us_country_list and cur_lang not in us_country_list[cur_alpha3_2]):
                            if cur_alpha3_1 in country_pair_syn_dict_dict_dict and cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                                del country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]
                        else:
                            valid_country_dict[cur_alpha3_2] = report_country_count_dict[cur_alpha3_2]

            oceania_gdp = []
            na_gdp = []
            sa_gdp = []
            global_gdp = []

            for i in range(len(country_syn_list)):
                if country_syn_dict["Continent"][i] == "Oceania":
                    oceania_gdp.append(country_syn_dict["2020_gdp"][i])
                elif country_syn_dict["Continent"][i] == "North America":
                    na_gdp.append(country_syn_dict["2020_gdp"][i])
                elif country_syn_dict["Continent"][i] == "South America":
                    sa_gdp.append(country_syn_dict["2020_gdp"][i])

                global_gdp.append(country_syn_dict["2020_gdp"][i])

            print("average gdp of oceania:", np.mean(oceania_gdp))
            print("average gdp of north america:", np.mean(na_gdp))
            print("average gdp of south america:", np.mean(sa_gdp))
            print("global average gdp: ", np.mean(global_gdp))

            print("country coverage number list:")
            for cur_alpha3_2 in valid_country_dict:
                print(cur_alpha3_2, "  ", valid_country_dict[cur_alpha3_2])

            inter_country_train_data = []
            inter_country_train_label = []
            fitlered_country_pair_num = 0
            fitlered_sampling_pair_num = 0

            country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
            for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                    cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                    if cur_country_pair['Country1'] == 'Nigeria':
                        special_country_idx = len(inter_country_train_data)
                        print("cur_country_idx:", special_country_idx)

                    cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                    cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                    cur_country_pair_sim = cur_country_pair["sim"]
                    cur_country_pair_same_lang = cur_country_pair["same_lang"]
                    cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                    cur_country_pair.pop('simple_lang_vec')
                    cur_country_pair.pop('lang_vec')
                    cur_country_pair.pop('sim')
                    cur_country_pair.pop('same_lang')
                    cur_country_pair.pop('same_spec_lang')


                    if np.isnan(cur_country_pair['border']):
                        continue
                    if np.isnan(cur_country_pair['diplomatic']):
                        continue
                    if np.isnan(cur_country_pair['investment_2018']):
                        continue
                    # if np.isnan(cur_country_pair['investment_2005']):
                    #     continue
                    # if np.isnan(cur_country_pair['investment_2008']):
                    #     continue
                    if np.isnan(cur_country_pair['trade']):
                        continue
                    if np.isnan(cur_country_pair['immgration']):
                        continue

                    for attr in cur_country_pair.values():
                        if isinstance(attr, int) or isinstance(attr, float):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                        # if isinstance(attr, list) and len(attr) == 1:
                        if isinstance(attr, list):
                            country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                    # feature_combo_list = []
                    # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                    #     temp_list = []
                    #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                    #         temp_list.append(c)
                    #     feature_combo_list.extend(temp_list)

                    try:
                        cur_train_data_row = [cur_country_pair['death_num']] + [cur_country_pair['border']] + cur_country_pair['continent_vec'] + [cur_country_pair['continent_sim']] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['gdp_binary']] + cur_country_pair['gdp_vec'] + [cur_country_pair['gini_binary']] + cur_country_pair['gini_index_vec'] + [cur_country_pair['dem_binary']] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['diplomatic']] + [cur_country_pair['investment_2018']] + [cur_country_pair['trade']] + [cur_country_pair['immgration']] + cur_country_pair_simple_lang_vec
                        inter_country_train_data.append(cur_train_data_row)

                        inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                    except:
                        traceback.print_exc()
                        fitlered_country_pair_num += 1

            print("fitlered_country_pair_num: ", fitlered_country_pair_num)


            # normalization
            minmax = MinMaxScaler()

            # using absolute value to simplify the trading effect
            inter_country_train_data = np.abs(inter_country_train_data)
            inter_country_train_data = list(inter_country_train_data)

            inter_country_train_data = minmax.fit_transform(inter_country_train_data)
            print("special country data: ", inter_country_train_data[special_country_idx])

            print("country pairs total number:", len(inter_country_train_label))

            inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
            inter_country_train_data_df.columns = ["Death num", "Neighbor", "Asia", "Europe", "Africa", "Oceania", "North America", "South America", "Antarctica", "Continent sim", "nato","eunion","brics", 'five_eyes', \
                                               "GDP(high)", "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                               "Gini Index(high)", "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                               "Democracy index(high)", "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                               "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                               "Same language"]

            oceania_inter_country_train_data_df = inter_country_train_data_df[inter_country_train_data_df["Oceania"] == 1]
            na_inter_country_train_data_df = inter_country_train_data_df[inter_country_train_data_df["North America"] == 1]

            # 0 is the pair number, 1 is the average similarity
            # y = inter_country_train_data_df[1]
            y = inter_country_train_label
            X = inter_country_train_data_df

            X_feature_names = X.columns.values.tolist()
            valid_X_feature_names = []
            # filter the columns with only 1 kind of value, e.g. all 0 or all 1
            for cur_feature in X_feature_names:
                multivalue_sign = 0

                cur_feature_values = X[cur_feature]
                cmp_value = cur_feature_values[0]
                for cur_feature_value in cur_feature_values:
                    if cur_feature_value != cmp_value:
                        multivalue_sign = 1
                        break
                if multivalue_sign == 1:
                    valid_X_feature_names.append(cur_feature)

            X = X[valid_X_feature_names]
            print("Shape of X:", X.shape)

            model = sm.OLS(y, sm.add_constant(X)).fit()
            model_sets.append(model)

            model_sets_res = summary_col(model_sets, model_names = cur_lang_countries,stars=True, float_format='%0.4f', info_dict={'Observation Num': nobs})
            cur_save_folder = f"results/disaster_lang_select_summary/{cur_lang}/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_lang + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())


            print("positive samples of each country:")
            print(report_country_count_dict)

    if args.option == "rm_pair_country_summary":
        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass


        # investment flow, using absolute values in the summary
        country_pair_investment = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
        country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in
                                        country_pair_investment.keys()}
        country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
                country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
            except:
                pass


        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass



        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass


        cur_lang_set = [args.data[-2:]]
        for cur_lang in cur_lang_set:
            model_sets = []
            cur_lang_countries = []
            valid_pair_count_dict = {}
            for select_country in top_country_list:
                country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
                # only consider the models of countries which speak this language as official language
                if cur_lang not in top_country_list[select_country]:
                    continue

                cur_lang_countries.append(select_country)
                print(select_country)

                cur_valid_pair_count = 0

                for i in range(len(country_syn_list)):
                    for j in range(len(country_syn_list)):
                        if i == j:
                            continue
                        else:
                            cur_idx1 = i
                            cur_idx2 = j

                            cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                            cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                            '''limit to usa news converage to global disasters'''
                            if cur_alpha3_2 != select_country:
                                continue

                            if (cur_alpha3_1 not in disaster_country_pair_stats_total) or (cur_alpha3_2 not in disaster_country_pair_stats_total[cur_alpha3_1]):
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = 0
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_pair_stats_total[cur_alpha3_1][cur_alpha3_2]
                                cur_valid_pair_count += 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = country_syn_dict["Country"][cur_idx1]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = country_syn_dict["Country"][cur_idx2]

                            if (cur_alpha3_1 in country_syn_dict["borders"][cur_idx2]) or (cur_alpha3_2 in country_syn_dict["borders"][cur_idx1]):
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['border'] = 0

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['dist'] = geopy.distance.geodesic((country_syn_dict["Latitude (average)"][cur_idx1],country_syn_dict["Longitude (average)"][cur_idx1]),(country_syn_dict["Latitude (average)"][cur_idx2],country_syn_dict["Longitude (average)"][cur_idx2])).km

                            # continent similarity
                            if country_syn_dict["Continent"][cur_idx1] == country_syn_dict["Continent"][cur_idx2]:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['continent_sim'] = 0

                            # Region1 similarity
                            if country_syn_dict["Region1"][cur_idx1] == country_syn_dict["Region1"][cur_idx2]:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 1
                            else:
                                country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]['region1_sim'] = 0

                            # language
                            try:
                                common_official_langs = list(set(country_syn_dict["Official language"][cur_idx1]).intersection(set(country_syn_dict["Official language"][cur_idx2])))
                            except:
                                common_official_langs = []
                            try:
                                common_official_langs_family = list(set(country_syn_dict["Official language family"][cur_idx1]).intersection(set(country_syn_dict["Official language family"][cur_idx2])))
                            except:
                                common_official_langs_family = []

                            simple_lang_vector = [0 for i in range(1)]
                            if common_official_langs != []:
                                simple_lang_vector[0] = 1
                            # elif common_official_langs_family != []:
                            #     simple_lang_vector[1] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_lang_vec"] = simple_lang_vector


                            lang_vector = [0 for i in range(10)]
                            if common_official_langs != []:
                                lang_full_name_map_keys = list(LANG_FULL_NAME_MAP.keys())
                                for lang in common_official_langs:
                                    if lang in lang_full_name_map_keys:
                                        lang_vector[lang_full_name_map_keys.index(lang)] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_vec"] = lang_vector

                            # political group categories, 4 kinds
                            common_group = []
                            for group in ['nato', 'eunion', 'brics']:
                                if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                    common_group.append(group)

                            across_group = []
                            if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['eunion'][
                                cur_idx2] == 'eunion') or (
                                    country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['nato'][
                                cur_idx2] == 'nato'):
                                across_group.append('nato-eunion')
                            if (country_syn_dict['nato'][cur_idx1] == 'nato' and country_syn_dict['brics'][
                                cur_idx2] == 'brics') or (
                                    country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['nato'][
                                cur_idx2] == 'nato'):
                                across_group.append('nato-brics')
                            if (country_syn_dict['eunion'][cur_idx1] == 'eunion' and country_syn_dict['brics'][
                                cur_idx2] == 'brics') or (
                                    country_syn_dict['brics'][cur_idx1] == 'brics' and country_syn_dict['eunion'][
                                cur_idx2] == 'eunion'):
                                across_group.append('eunion-brics')

                            simple_political_group_vector = [0]
                            if common_group != []:
                                simple_political_group_vector[0] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["simple_political_group_vec"] = simple_political_group_vector

                            '''political group featrues'''
                            political_group_vector = [0 for i in range(3)]
                            if 'nato' in common_group:
                                political_group_vector[0] = 1
                            elif 'eunion' in common_group:
                                political_group_vector[1] = 1
                            elif 'brics' in common_group:
                                political_group_vector[2] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                                "political_group_vec"] = political_group_vector

                            across_group_vector = [0 for i in range(3)]
                            if 'nato-eunion' in across_group:
                                across_group_vector[0] = 1
                            if 'nato-brics' in across_group:
                                across_group_vector[1] = 1
                            if 'eunion-brics' in across_group:
                                across_group_vector[2] = 1

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["across_group_vec"] = across_group_vector

                            # unitary type combination categories, 6 kinds
                            simple_unitary_vector = [0]
                            if country_syn_dict["Unitary"][cur_idx1] == country_syn_dict["Unitary"][cur_idx2]:
                                simple_unitary_vector[0] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                                "simple_unitary_vector"] = simple_unitary_vector

                            unitary_vector = [0 for i in range(6)]
                            combo_unitary_types = [country_syn_dict["Unitary"][cur_idx1], country_syn_dict["Unitary"][cur_idx2]]
                            if combo_unitary_types == ["unitary_republics", "unitary_republics"]:
                                unitary_vector[0] = 1
                            elif combo_unitary_types == ["federalism", "federalism"]:
                                unitary_vector[1] = 1
                            # elif ("unitary_republics" not in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                            #     unitary_vector[2] = 1
                            elif ("unitary_republics" in combo_unitary_types) and ("federalism" in combo_unitary_types):
                                unitary_vector[3] = 1
                            elif ("unitary_republics" in combo_unitary_types) and ("federalism" not in combo_unitary_types):
                                unitary_vector[4] = 1
                            elif ("unitary_republics" not in combo_unitary_types) and ("federalism" in combo_unitary_types):
                                unitary_vector[5] = 1
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                            gdp_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["2020_gdp"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["2020_gdp"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["2020_gdp"][cur_idx1] < 500000 * 1000000:
                                    gdp_idx_sign1 = 0
                                else:
                                    gdp_idx_sign1 = 1
                                if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                    gdp_idx_sign2 = 0
                                else:
                                    gdp_idx_sign2 = 1

                                if gdp_idx_sign1 == gdp_idx_sign2:
                                    gdp_converg = 1

                                    if gdp_idx_sign1 == 0:
                                        # take this class as reference
                                        # gdp_vec[0] = 1
                                        pass
                                    else:
                                        gdp_vec[1] = 1
                                else:
                                    gdp_converg = 0
                                    if gdp_idx_sign1 == 0:
                                        gdp_vec[2] = 1
                                    else:
                                        gdp_vec[3] = 1

                                # confine dummy vector for the regression model of a certain country to have the reference class
                                # e.g. if the country is 'high', there will only be "low-high" and "high-high",
                                # treat "low-high" as the reference class,
                                # because for some country they don't have enough disaster or conflicts matching to "low" countries

                                if country_syn_dict["2020_gdp"][cur_idx2] < 500000 * 1000000:
                                    gdp_vec[0] = 0
                                else:
                                    gdp_vec[2] = 0

                            else:
                                gdp_converg = float("nan")
                                gdp_vec = [float("nan") for i in range(4)]

                            # for test bug
                            if gdp_vec[0] == 1 or gdp_vec[3] == 1:
                                print()

                            gini_index_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["gini_index"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["gini_index"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["gini_index"][cur_idx1] < 35:
                                    gini_index_idx_sign1 = 0
                                else:
                                    gini_index_idx_sign1 = 1
                                if country_syn_dict["gini_index"][cur_idx2] < 35:
                                    gini_index_idx_sign2 = 0
                                else:
                                    gini_index_idx_sign2 = 1

                                if gini_index_idx_sign1 == gini_index_idx_sign2:
                                    gini_index_converg = 1

                                    if gini_index_idx_sign1 == 0:
                                        # take this class as reference
                                        # gini_index_vec[0] = 1
                                        pass
                                    else:
                                        gini_index_vec[1] = 1
                                else:
                                    gini_index_converg = 0

                                    if gini_index_idx_sign1 == 0:
                                        gini_index_vec[2] = 1
                                    else:
                                        gini_index_vec[3] = 1

                                if country_syn_dict["gini_index"][cur_idx2] < 35:
                                    gini_index_vec[0] = 0
                                else:
                                    gini_index_vec[2] = 0
                            else:
                                gini_index_converg = float("nan")
                                gini_index_vec = [float("nan") for i in range(4)]

                            dem_idx_vec = [0 for i in range(4)]  # 'low-low','high-high','low->high', "high->low"
                            if not np.isnan(country_syn_dict["eiu"][cur_idx1]) and not np.isnan(
                                    country_syn_dict["eiu"][cur_idx2]):
                                # 0 stands for low, 1 stands for high
                                if country_syn_dict["eiu"][cur_idx1] < 5:
                                    dem_idx_sign1 = 0
                                else:
                                    dem_idx_sign1 = 1
                                if country_syn_dict["eiu"][cur_idx2] < 5:
                                    dem_idx_sign2 = 0
                                else:
                                    dem_idx_sign2 = 1

                                if dem_idx_sign1 == dem_idx_sign2:
                                    dem_idx_converg = 1

                                    if dem_idx_sign1 == 0:
                                        # take this class as reference
                                        # dem_idx_vec[0] = 1
                                        pass
                                    else:
                                        dem_idx_vec[1] = 1
                                else:
                                    dem_idx_converg = 0

                                    if dem_idx_sign1 == 0:
                                        dem_idx_vec[2] = 1
                                    else:
                                        dem_idx_vec[3] = 1

                                if country_syn_dict["eiu"][cur_idx2] < 5:
                                    dem_idx_vec[0] = 0
                                else:
                                    dem_idx_vec[2] = 0
                            else:
                                dem_idx_converg = float("nan")
                                dem_idx_vec = [float("nan") for i in range(4)]

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_converg"] = gdp_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gdp_vec"] = gdp_vec

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_converg"] = gini_index_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index_vec"] = gini_index_vec

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_converg"] = dem_idx_converg
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx_vec"] = dem_idx_vec

                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["diplomatic"] = country_pair_diplomatic_relation_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["investment"] = country_pair_investment_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["trade"] = country_pair_trade_value_dict[cur_alpha3_1][cur_alpha3_2]
                            country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["immgration"] = country_pair_immgration_value_dict[cur_alpha3_1][cur_alpha3_2]

                valid_pair_count_dict[select_country] = cur_valid_pair_count

                inter_country_train_data = []
                inter_country_train_label = []
                fitlered_country_pair_num = 0
                fitlered_sampling_pair_num = 0

                country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                    for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                        cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])

                        cur_country_pair_simple_lang_vec = cur_country_pair["simple_lang_vec"]
                        cur_country_pair_lang_vec = cur_country_pair["lang_vec"]
                        cur_country_pair_sim = cur_country_pair["sim"]
                        cur_country_pair_same_lang = cur_country_pair["same_lang"]
                        cur_country_pair_same_spec_lang = cur_country_pair["same_spec_lang"]

                        cur_country_pair.pop('simple_lang_vec')
                        cur_country_pair.pop('lang_vec')
                        cur_country_pair.pop('sim')
                        cur_country_pair.pop('same_lang')
                        cur_country_pair.pop('same_spec_lang')


                        if np.isnan(cur_country_pair['border']):
                            continue
                        if np.isnan(cur_country_pair['diplomatic']):
                            continue
                        if np.isnan(cur_country_pair['investment']):
                            continue
                        if np.isnan(cur_country_pair['trade']):
                            continue
                        if np.isnan(cur_country_pair['immgration']):
                            continue

                        for attr in cur_country_pair.values():
                            if isinstance(attr, int) or isinstance(attr, float):
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                            # if isinstance(attr, list) and len(attr) == 1:
                            if isinstance(attr, list):
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                        # feature_combo_list = []
                        # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                        #     temp_list = []
                        #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                        #         temp_list.append(c)
                        #     feature_combo_list.extend(temp_list)

                        try:

                            cur_train_data_row = [cur_country_pair['border']] + [cur_country_pair['continent_sim']] + cur_country_pair['gdp_vec'] + cur_country_pair['gini_index_vec'] + cur_country_pair['dem_idx_vec'] + [cur_country_pair['diplomatic']] + [cur_country_pair['investment']] + [cur_country_pair['trade']] + [cur_country_pair['immgration']] + cur_country_pair_simple_lang_vec
                            inter_country_train_data.append(cur_train_data_row)

                            inter_country_train_label.append(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2][0])

                        except:
                            traceback.print_exc()
                            fitlered_country_pair_num += 1
                print("fitlered_country_pair_num: ", fitlered_country_pair_num)


                # normalization
                minmax = MinMaxScaler()

                # using absolute value to simplify the trading effect
                inter_country_train_data = np.abs(inter_country_train_data)
                inter_country_train_data = list(inter_country_train_data)

                inter_country_train_data = minmax.fit_transform(inter_country_train_data)


                print("country pairs total number:", len(inter_country_train_label))

                inter_country_train_data_df = pd.DataFrame(inter_country_train_data)
                inter_country_train_data_df.columns = ["Neighbor", "Continent sim", \
                                                   "GDP(low-low)", "GDP(high-high)", "GDP(low-high)", "GDP(high-low)",\
                                                   "Gini Index(low-low)", "Gini Index(high-high)", "Gini Index(low-high)", "Gini Index(high-low)",\
                                                   "Democracy index(low-low)", "Democracy index(high-high)", "Democracy index(low-high)", "Democracy index(high-low)",\
                                                   "Diplomatic Relation", "Investment", "Trade", "Immigration", \
                                                   "Same language"]



                # 0 is the pair number, 1 is the average similarity
                # y = inter_country_train_data_df[1]
                y = inter_country_train_label
                X = inter_country_train_data_df

                X_feature_names = X.columns.values.tolist()
                valid_X_feature_names = []
                # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                for cur_feature in X_feature_names:
                    multivalue_sign = 0

                    cur_feature_values = X[cur_feature]
                    cmp_value = cur_feature_values[0]
                    for cur_feature_value in cur_feature_values:
                        if cur_feature_value != cmp_value:
                            multivalue_sign = 1
                            break
                    if multivalue_sign == 1:
                        valid_X_feature_names.append(cur_feature)

                X = X[valid_X_feature_names]
                print("Shape of X:", X.shape)
                model = sm.OLS(y, sm.add_constant(X)).fit()
                model_sets.append(model)

            model_sets_res = summary_col(model_sets, model_names = cur_lang_countries,stars=True, float_format='%0.4f', info_dict={'Observation Num': nobs})
            print("country:", select_country)
            cur_save_folder = f"results/disaster_country_pair_lang_summary/{cur_lang}/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_lang + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())


            print("positive samples of each country:")
            print(valid_pair_count_dict)

    if args.option == "rm_country":
        # load geographic info
        country_geography_list = pd.read_csv(data_route+"country_info/country_geo_location.csv")

        country_alpha3_code = country_geography_list["Alpha-3 code"].to_list()
        country_full_name = country_geography_list["Country"].to_list()
        country_latitude = country_geography_list["Latitude (average)"].to_list()
        country_longitude = country_geography_list["Longitude (average)"].to_list()

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route+"country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route+"country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route+"country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(data_route+"bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route+"country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route+"country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route+"country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        country_gdp_per_person_list = pd.read_csv(data_route+"country_info/country_gdp_per_person.csv")
        country_gdp_per_person_list = country_gdp_per_person_list[["Country Name", "2020"]]
        country_gdp_per_person_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_per_person_list.rename(columns={"2020": '2020_gdp_per_person'}, inplace=True)

        country_gdp_per_person_list = pd.merge(country_gdp_list, country_gdp_per_person_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route+"country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_per_person_list, country_gini_index_list, how='left',
                                           on='Alpha-3 code')

        '''load it user rate info'''
        country_it_user_list = pd.read_csv(data_route+"country_info/country_it_user.csv")
        country_it_user_list = country_it_user_list[["Country or area", "Pct"]]
        country_it_user_list.rename(columns={"Country or area": 'Country'}, inplace=True)
        country_it_user_list.rename(columns={"Pct": 'it_user_rate'}, inplace=True)
        country_it_user_list['Country'] = country_it_user_list.apply(lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_it_user_list['it_user_rate'] = country_it_user_list.apply(
            lambda x: float(x['it_user_rate'].replace("%", "")), axis=1)
        country_it_user_list = pd.merge(country_gini_index_list, country_it_user_list, how='left', on='Country')

        '''load literacy rate info'''
        country_literacy_list = pd.read_csv(data_route+"country_info/country_literacy_rate.csv")
        country_literacy_list = country_literacy_list[["cca3", "latestRate"]]
        country_literacy_list.rename(columns={"cca3": 'Alpha-3 code'}, inplace=True)
        country_literacy_list.rename(columns={"latestRate": 'literacy_rate'}, inplace=True)
        country_literacy_list = pd.merge(country_it_user_list, country_literacy_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route+"country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_literacy_list, country_peace_index_list, how='left', on='Country')

        '''load migration info'''
        country_migration_list = pd.read_csv(data_route+"country_info/migration/country_migration.csv")
        country_migration_list = country_migration_list[["Country Code", "2020"]]
        country_migration_list.rename(columns={"Country Code": 'Alpha-3 code'}, inplace=True)
        country_migration_list.rename(columns={"2020": 'net_migration'}, inplace=True)
        country_migration_list = pd.merge(country_peace_index_list, country_migration_list, how='left',
                                          on='Alpha-3 code')

        '''load religion distribution info'''
        country_religion_distribution_list = pd.read_csv(data_route+"country_info/religion/country_religion_distribution.csv")
        country_religion_distribution_dict = {key: country_religion_distribution_list[key].to_list() for key in
                                              country_religion_distribution_list.keys()}
        country_religion_distribution_dict_list = defaultdict(list)
        for country_idx in range(len(country_religion_distribution_dict["Country"])):
            cur_country = country_religion_distribution_dict["Country"][country_idx]
            cur_christian = float(country_religion_distribution_dict["CHRISTIAN"][country_idx])
            cur_muslim = country_religion_distribution_dict["MUSLIM"][country_idx]
            cur_unaffil = country_religion_distribution_dict["UNAFFIL"][country_idx]
            cur_hindu = country_religion_distribution_dict["HINDU"][country_idx]
            cur_buddhist = country_religion_distribution_dict["BUDDHIST"][country_idx]
            cur_folk_religion = country_religion_distribution_dict["FOLK RELIGION"][country_idx]
            cur_other_religion = country_religion_distribution_dict["OTHER RELIGION"][country_idx]
            cur_jewish = country_religion_distribution_dict["JEWISH"][country_idx]

            country_religion_distribution_dict_list[cur_country].append(cur_christian)
            country_religion_distribution_dict_list[cur_country].append(cur_muslim)
            country_religion_distribution_dict_list[cur_country].append(cur_unaffil)
            country_religion_distribution_dict_list[cur_country].append(cur_hindu)
            country_religion_distribution_dict_list[cur_country].append(cur_buddhist)
            country_religion_distribution_dict_list[cur_country].append(cur_folk_religion)
            country_religion_distribution_dict_list[cur_country].append(cur_other_religion)
            country_religion_distribution_dict_list[cur_country].append(cur_jewish)

        country_religion_distribution_entropy_dict = {}
        for cur_country in country_religion_distribution_dict_list:
            country_religion_distribution_entropy_dict[cur_country] = (
                        1 - entropy(country_religion_distribution_dict_list[cur_country]) / np.log(
                    len(country_religion_distribution_dict_list[cur_country])))
        country_religion_distribution_list["religion_entropy"] = country_religion_distribution_list.apply(
            lambda x: country_religion_distribution_entropy_dict[x["Country"]], axis=1)
        country_religion_distribution_list = country_religion_distribution_list[["Country", "religion_entropy"]]
        country_religion_distribution_list = pd.merge(country_migration_list, country_religion_distribution_list,how='left', on='Country')

        '''load population and area info'''
        country_pop_area_list = pd.read_csv(data_route+"country_info/country_population_area.csv")
        country_pop_area_list = country_pop_area_list[["cca3", "pop2021", "area", "density"]]
        country_pop_area_list.rename(columns={"cca3": 'Alpha-3 code'}, inplace=True)
        country_pop_area_list = pd.merge(country_religion_distribution_list, country_pop_area_list, how='left', on='Alpha-3 code')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_pop_area_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route+f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        '''filter the countries without corresponding data'''
        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", "2020_gdp", "gini_index", "it_user_rate", "literacy_rate", "peace_index", "net_migration", "religion_entropy", "pop2021", "area"])

        country_pair_syn_dict_dict_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}

        for cur_disaster_type in disaster_country_stats:
            for i in range(len(country_syn_list)):
                for j in range(len(country_syn_list)):
                    if i == j:
                        cur_idx1 = i
                        cur_idx2 = i

                        # a = country_syn_dict["Alpha-3 code"]
                        # b = a[0]
                        # c = a[1]

                        cur_alpha3_1 = country_syn_dict["Alpha-3 code"][cur_idx1]
                        cur_alpha3_2 = country_syn_dict["Alpha-3 code"][cur_idx2]

                        # print(cur_idx1, cur_alpha3_1, cur_idx2, cur_alpha3_2)

                        if cur_alpha3_1 not in disaster_country_stats[cur_disaster_type]:
                            continue

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"] = disaster_country_stats[cur_disaster_type][cur_alpha3_1]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country1"] = \
                        country_syn_dict["Country"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["Country2"] = \
                        country_syn_dict["Country"][cur_idx2]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][""]

                        # language categories
                        # About 12 categorical values = 10 languages (or less, because some languages like Polish are official in only one country) + “Not the same language but same family” + “Not the same language and different family” categories
                        try:
                            common_official_langs = list(country_syn_dict["Official language"][cur_idx1])
                        except:
                            common_official_langs = []
                        try:
                            common_official_langs_family = list(country_syn_dict["Official language family"][cur_idx1])
                        except:
                            common_official_langs_family = []

                        lang_num_vector = [len(common_official_langs), len(common_official_langs_family)]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["lang_num_vec"] = lang_num_vector

                        # political group categories, 4 kinds
                        common_group = []
                        for group in ['nato', 'eunion', 'brics']:
                            if country_syn_dict[group][cur_idx1] == country_syn_dict[group][cur_idx2]:
                                common_group.append(group)

                        ''' not disentangled political group features'''
                        political_group_vector = [0 for i in range(3)]
                        if ('nato' in common_group):
                            political_group_vector[0] = 1
                        elif ('eunion' in common_group):
                            political_group_vector[1] = 1
                        elif ('brics' in common_group):
                            political_group_vector[2] = 1

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2][
                            "political_group_vec"] = political_group_vector

                        # unitary type combination categories, 6 kinds
                        unitary_vector = [0 for i in range(2)]
                        cur_unitary_type = country_syn_dict["Unitary"][cur_idx1]
                        if cur_unitary_type == "unitary_republics":
                            unitary_vector[0] = 1
                        elif cur_unitary_type == "federalism":
                            unitary_vector[1] = 1
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["unitary_vec"] = unitary_vector

                        # democracy index, press freedom index, gdp, gdp per person, population, area, density

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["dem_idx"] = country_syn_dict["eiu"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["pf_idx"] = country_syn_dict["Press Freedom"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_gdp"] = \
                        country_syn_dict["2020_gdp"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_gdp_per_person"] = country_syn_dict["2020_gdp_per_person"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["gini_index"] = \
                        country_syn_dict["gini_index"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["it_user_rate"] = \
                        country_syn_dict["it_user_rate"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["literacy_rate"] = \
                        country_syn_dict["literacy_rate"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["peace_index"] = \
                        country_syn_dict["peace_index"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["net_migration"] = \
                        country_syn_dict["net_migration"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["religion_entropy"] = \
                        country_syn_dict["religion_entropy"][cur_idx1]

                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["2020_pop"] = country_syn_dict["pop2021"][cur_idx1]
                        country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["area"] = country_syn_dict["area"][cur_idx1]
                        # country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["density"] = country_syn_dict["density"][cur_idx1]


            model_sets = []
            for search_metric in ['vif', 'aic']:
                intra_country_train_data = []
                intra_country_train_label = []

                country_pair_syn_dict_dict_list = defaultdict(lambda: defaultdict(list))
                for cur_alpha3_1 in country_pair_syn_dict_dict_dict:
                    for cur_alpha3_2 in country_pair_syn_dict_dict_dict[cur_alpha3_1]:
                        cur_country_pair = copy.deepcopy(country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2])
                        cur_country_pair_sim = country_pair_syn_dict_dict_dict[cur_alpha3_1][cur_alpha3_2]["disaster_art_count"]

                        # cur_country_pair.pop('disaster_art_count')



                        # if cur_country_pair['pair_count'] < min_pair_count:
                        #     continue
                        # if np.isnan(cur_country_pair['dem_idx']):
                        #     continue
                        # if np.isnan(cur_country_pair['pf_idx']):
                        #     continue
                        # if np.isnan(cur_country_pair['2020_gdp']):
                        #     continue
                        # if np.isnan(cur_country_pair['2020_gdp_per_person']):
                        #     continue
                        for attr in cur_country_pair.values():
                            if isinstance(attr, int) or isinstance(attr, float):
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2].append(attr)
                            # if isinstance(attr, list) and len(attr) == 1:
                            if isinstance(attr, list) and len(attr) > 0:
                                country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2] += attr

                        # feature_combo_list = []
                        # for i in range(len(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2])):
                        #     temp_list = []
                        #     for c in combinations(country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2],i):
                        #         temp_list.append(c)
                        #     feature_combo_list.extend(temp_list)

                        cur_train_data_row = country_pair_syn_dict_dict_list[cur_alpha3_1][cur_alpha3_2]
                        intra_country_train_data.append(cur_train_data_row)
                        intra_country_train_label.append(cur_country_pair_sim)

                # normalization
                minmax = MinMaxScaler()
                intra_country_train_data = minmax.fit_transform(intra_country_train_data)
                print("country pairs total number:", len(intra_country_train_label))

                intra_country_train_data_df = pd.DataFrame(intra_country_train_data)

                intra_country_train_data_df.columns = ['disaster_art_count',  "Number of language",
                                                       "Number of language family", \
                                                       "Within NATO?", "Within BRICS?", "Within EU?", "Republic?",
                                                       "Federalism?",  \
                                                       "Democracy index", "GDP", "Gini index", "IT User Rate",
                                                       "Literacy Rate", "Peace Index", "Net Migration",
                                                       "Religion Distribution Entropy", "Population", "Area"]

                # 0 is the pair number, 1 is the average similarity
                y = intra_country_train_label
                X = intra_country_train_data_df

                X_feature_names = X.columns.values.tolist()
                valid_X_feature_names = []
                # filter the columns with only 1 kind of value, e.g. all 0 or all 1
                for cur_feature in X_feature_names:
                    multivalue_sign = 0

                    cur_feature_values = X[cur_feature]
                    cmp_value = cur_feature_values[0]
                    for cur_feature_value in cur_feature_values:
                        if cur_feature_value != cmp_value:
                            multivalue_sign = 1
                            break
                    if multivalue_sign == 1:
                        valid_X_feature_names.append(cur_feature)

                X = X[valid_X_feature_names]

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

                # Create an empty dictionary that will be used to store our results
                function_dict = {'predictor': [], 'r-squared': []}
                # Iterate through every column in X
                cols = list(X.columns)
                for col in cols:
                    # Create a dataframe called selected_X with only the 1 column
                    selected_X = X[[col]]
                    # Fit a model for our target and our selected column
                    model = sm.OLS(y, sm.add_constant(selected_X)).fit()
                    # Predict what our target would be for our model
                    y_preds = model.predict(sm.add_constant(selected_X))
                    # Add the column name to our dictionary
                    function_dict['predictor'].append(col)
                    # Calculate the r-squared value between the target and predicted target
                    r2 = np.corrcoef(y, y_preds)[0, 1] ** 2
                    # Add the r-squared value to our dictionary
                    function_dict['r-squared'].append(r2)

                # Once it's iterated through every column, turn our dictionary into a DataFrame and sort it
                function_df = pd.DataFrame(function_dict).sort_values(by=['r-squared'], ascending=False)
                # Display only the top 5 predictors

                if search_metric == "vif":
                    selected_features = [function_df['predictor'].iat[0]]
                    features_to_ignore = []

                    # Since our function's ignore_features list is already empty, we don't need to
                    # include our features_to_ignore list.
                    while len(selected_features) + len(features_to_ignore) < len(cols):
                        next_feature = next_possible_feature(X_npf=X, y_npf=y, current_features=selected_features,
                                                             ignore_features=features_to_ignore)[0]
                        # check vif score
                        vif_factor = 5
                        temp_selected_features = selected_features + [next_feature]
                        temp_X = X[temp_selected_features]
                        temp_vif = pd.DataFrame()
                        temp_vif["features"] = temp_X.columns
                        temp_vif["VIF"] = [variance_inflation_factor(temp_X.values, i) for i in range(len(temp_X.columns))]
                        cur_vif = temp_vif["VIF"].iat[-1]
                        if cur_vif <= vif_factor:
                            selected_features = temp_selected_features
                        else:
                            features_to_ignore.append(next_feature)
                elif search_metric == "aic":
                    selected_features = [function_df['predictor'].iat[0]]
                    rest_features = []
                    for cur_feature in valid_X_feature_names:
                        if cur_feature not in selected_features:
                            rest_features.append(cur_feature)

                    best_aic = 10000000
                    search_max_time = 10000
                    search_time = 0
                    while len(selected_features) < len(cols) or search_time >= search_max_time:
                        # if there is no change in this turn then meaning no feature is selected.
                        # Should also stop search in this case
                        change_sign = 0
                        temp_feature_sets = [selected_features + [temp_feature] for temp_feature in rest_features]
                        for temp_feature_set in temp_feature_sets:
                            temp_X = X[temp_feature_set]
                            temp_model = sm.OLS(y, sm.add_constant(temp_X)).fit()
                            if temp_model.aic < best_aic:
                                best_aic = temp_model.aic
                                selected_features = temp_feature_set
                                change_sign = 1
                        if change_sign == 0:
                            break
                        search_time += 1

                model = sm.OLS(y, sm.add_constant(X[selected_features])).fit()
                model_sets.append(model)


                # print(model.summary())
                # print(model.summary().as_latex())

            model_sets_res = summary_col(model_sets, stars=True)

            print("disaster: ", cur_disaster_type)
            cur_save_folder = f"results/disaster_country/{args.data}/"
            if not os.path.exists(cur_save_folder):
                os.makedirs(cur_save_folder)

            with open(f"{cur_save_folder}/" + cur_disaster_type + '.txt', 'w') as f:
                f.write(str(model_sets_res))
            print(model_sets_res.as_latex())

    if args.option == 'plot':
        country_geography_list = pd.read_csv(data_route + "country_info/country_geo_location.csv")

        # filtering the intra-country pairs whose country official language in not covered in the 10 languages of our annotation
        country_official_languages = pd.read_csv(data_route + "country_info/country_official_language.csv")
        country_official_languages["Country"] = country_official_languages.apply(
            lambda x: x['Country'].replace("\xa0", ""), axis=1)
        country_official_languages["Official language"] = country_official_languages.apply(
            lambda x: x['Official language'].replace("\xa0", "").replace("\u2028", "\n").split("\n"), axis=1)
        country_official_languages["Official language family"] = country_official_languages.apply(
            lambda x: [LANG_FAMILY[LANG_FULL_NAME_MAP[lang]] for lang in x['Official language'] if
                       lang in LANG_FULL_NAME_MAP], axis=1)

        country_official_languages = pd.merge(country_geography_list, country_official_languages, how='left',
                                              on='Country')

        '''load country neighbor/border info'''
        country_neighbors = pd.read_csv(data_route + "country_info/country_neighbors.csv")
        country_neighbors["name"] = country_neighbors.apply(lambda x: x["name"].split(",")[0], axis=1)
        country_neighbors["borders"] = country_neighbors.apply(
            lambda x: x["borders"].split(",") if isinstance(x["borders"], str) else [], axis=1)
        country_neighbors = country_neighbors.drop(
            ["status", "currencies", "capital", "region", "subregion", "languages", "latlng", "area", "demonyms"],
            axis=1)
        country_neighbors.rename(columns={'name': 'Country'}, inplace=True)
        country_neighbors = pd.merge(country_official_languages, country_neighbors, how='left', on='Country')

        '''load continent info'''
        country_continent = pd.read_csv(data_route + "country_info/country_continent.csv")
        country_continent = pd.merge(country_neighbors, country_continent, how='left', on='Country')

        '''load democracy index info'''
        country_democracy_index_list = pd.read_csv(
            data_route + "bias_dataset/2019_democracy_index/2019_democracy_index.csv")
        country_democracy_index_list = country_democracy_index_list[country_democracy_index_list["year"] == 2019]
        country_democracy_index_list.rename(columns={'country': 'Country'}, inplace=True)
        country_democracy_index_list = pd.merge(country_continent, country_democracy_index_list, how='left',
                                                on='Country')

        '''load press freedom index info'''
        country_press_freedom_index_list = pd.read_csv(data_route + "country_info/country_press_freedom_index.csv")
        country_press_freedom_index_list = pd.merge(country_democracy_index_list, country_press_freedom_index_list,
                                                    how='left', on='Country')

        '''load unitary state info'''
        country_unitary_state_list = pd.read_csv(data_route + "country_info/country_unitary_state.csv")
        country_unitary_state_list = pd.merge(country_press_freedom_index_list, country_unitary_state_list, how='left',
                                              on='Country')

        '''load gdp info'''
        country_gdp_list = pd.read_csv(data_route + "country_info/country_gdp.csv")
        country_gdp_list = country_gdp_list[["Country Name", "2020"]]
        country_gdp_list.rename(columns={"Country Name": 'Country'}, inplace=True)
        country_gdp_list.rename(columns={"2020": '2020_gdp'}, inplace=True)

        country_gdp_list = pd.merge(country_unitary_state_list, country_gdp_list, how='left', on='Country')

        '''load gini index info'''
        country_gini_index_list = pd.read_csv(data_route + "country_info/country_gini_index.csv")
        country_gini_index_list = pd.merge(country_gdp_list, country_gini_index_list, how='left', on='Alpha-3 code')

        '''load peace index info'''
        country_peace_index_list = pd.read_csv(data_route + "country_info/country_peace_index.csv")
        country_peace_index_list = country_peace_index_list[["Country", "2020 Rate"]]
        country_peace_index_list.rename(columns={"2020 Rate": 'peace_index'}, inplace=True)
        # to align the sign with other factors
        country_peace_index_list['Country'] = country_peace_index_list.apply(lambda x: x['Country'].replace("\xa0", ""),
                                                                             axis=1)
        country_peace_index_list['peace_index'] = country_peace_index_list.apply(lambda x: -x['peace_index'], axis=1)
        country_peace_index_list = pd.merge(country_gini_index_list, country_peace_index_list, how='left', on='Country')

        '''load political group info'''
        # dict for speedy query
        # one country or country pair can belong to multiple political group
        country_syn_list = country_gini_index_list

        political_group_df_dict = {}
        political_group_alpha3_dict = {}
        for group in political_group_list:
            political_group_df_dict[group] = pd.read_csv(data_route + f"country_info/political_group/{group}.csv")
            # nato csv contains some other countries
            if group == "nato":
                political_group_df_dict["nato"] = political_group_df_dict["nato"][
                    political_group_df_dict["nato"]["Category"] == "NATO"]

            political_group_df_dict[group].rename(columns={'Alpha-3': 'Alpha-3 code'}, inplace=True)
            political_group_df_dict[group] = political_group_df_dict[group][['Alpha-3 code']]
            political_group_df_dict[group] = political_group_df_dict[group].drop_duplicates()
            political_group_alpha3_dict[group] = {alpha3: 1 for alpha3 in
                                                  political_group_df_dict[group]['Alpha-3 code'].to_list()}

            political_group_df_dict[group][group] = political_group_df_dict[group].apply(lambda x: group, axis=1)
            country_syn_list = pd.merge(country_syn_list, political_group_df_dict[group], how='left', on='Alpha-3 code')

        country_syn_list = country_syn_list.dropna(subset=["eiu", "Press Freedom", '2020_gdp', "gini_index"])

        # plot each factor as a bar plot
        for column in country_syn_list.columns:
            try:
                if isinstance(list(country_syn_list[column])[0], str):
                    continue
                plt.figure()  # Create a new figure
                plt.bar(country_syn_list.index, country_syn_list[column], label=column)
                plt.title(f'{column} of countries')
                plt.xlabel('Countries')
                plt.ylabel(column)
                plt.legend()

                # Save the figure
                plt.savefig(f'results/figure/factors/{column}_bar_plot.png')  # Save as PNG
                plt.close()  # Close the figure to free up memory
            except:
                pass

        country_syn_dict = {key: country_syn_list[key].to_list() for key in country_syn_list.keys()}
        check_lang_country = []
        for border_idx in range(len(country_syn_dict['borders'])):
            if not isinstance(country_syn_dict['borders'][border_idx], list):
                country_syn_dict['borders'][border_idx] = []
        country_to_alpha3_dict = {}
        alpha2_to_alpha3_dict = {}
        for i in range(len(country_syn_dict["Country"])):
            country_to_alpha3_dict[country_syn_dict["Country"][i]] = country_syn_dict["Alpha-3 code"][i]
            alpha2_to_alpha3_dict[country_syn_dict["Alpha-2 code"][i]] = country_syn_dict["Alpha-3 code"][i]

        # diplomatic relation flow
        country_pair_diplomatic_relation = pd.read_csv(
            data_route + "country_info/diplomatic_relation/Diplometrics Diplomatic Representation 1960-2020_20211215.csv")
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            country_pair_diplomatic_relation["Year"] == 2020]
        country_pair_diplomatic_relation = country_pair_diplomatic_relation[
            ["Destination", "Sending Country", "Embassy"]]

        country_pair_diplomatic_relation_dict = {key: country_pair_diplomatic_relation[key].to_list() for key in
                                                 country_pair_diplomatic_relation.keys()}
        country_pair_diplomatic_relation_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_diplomatic_relation_dict["Destination"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Destination"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[
                    country_pair_diplomatic_relation_dict["Sending Country"][country_idx]]

                country_pair_diplomatic_relation_value_dict[cur_destination_alpha3][cur_sending_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
                country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3] = \
                country_pair_diplomatic_relation_dict["Embassy"][country_idx]
            except:
                pass

        # plot diplomatic_relation bar plot
        cur_factor = 'diplomatic_relation'
        country_pair_diplomatic_relation_dict_for_plot = []
        for cur_sending_alpha3 in country_pair_diplomatic_relation_value_dict:
            for cur_destination_alpha3 in country_pair_diplomatic_relation_value_dict[cur_sending_alpha3]:
                country_pair_diplomatic_relation_dict_for_plot.append(country_pair_diplomatic_relation_value_dict[cur_sending_alpha3][cur_destination_alpha3])
        plt.figure()  # Create a new figure
        plt.bar(range(len(country_pair_diplomatic_relation_dict_for_plot)), country_pair_diplomatic_relation_dict_for_plot, label=cur_factor)
        plt.title(f'{cur_factor} of countries')
        plt.xlabel('Countries pairs')
        plt.ylabel(cur_factor)
        plt.legend()
        # Save the figure
        plt.savefig(f'results/figure/factors/{cur_factor}_bar_plot.png')  # Save as PNG
        plt.close()

        # investment flow, using absolute values in the summary
        country_pair_investment = pd.read_csv(data_route + "country_info/economy_flow_raw_data/country_investment_flow.csv")
        country_pair_investment = country_pair_investment[country_pair_investment["INDICATOR"] == "VALUE"]
        country_pair_investment = country_pair_investment[["REPORT_CTRY", "PARTNER_CTRY", "2018"]]

        country_pair_investment_dict = {key: country_pair_investment[key].to_list() for key in
                                        country_pair_investment.keys()}
        country_pair_investment_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_investment_dict["REPORT_CTRY"])):
            try:
                cur_destination_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["REPORT_CTRY"][country_idx]]
                cur_sending_alpha3 = alpha2_to_alpha3_dict[country_pair_investment_dict["PARTNER_CTRY"][country_idx]]

                country_pair_investment_value_dict[cur_destination_alpha3][cur_sending_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
                country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3] += \
                country_pair_investment_dict["2018"][country_idx]
            except:
                pass

        # plot investment bar plot
        cur_factor = 'investment'
        country_pair_investment_dict_for_plot = []
        for cur_sending_alpha3 in country_pair_investment_value_dict:
            for cur_destination_alpha3 in country_pair_investment_value_dict[cur_sending_alpha3]:
                country_pair_investment_dict_for_plot.append(country_pair_investment_value_dict[cur_sending_alpha3][cur_destination_alpha3])
        plt.figure()  # Create a new figure
        plt.ylim(-150000, 150000)
        plt.bar(range(len(country_pair_investment_dict_for_plot)), country_pair_investment_dict_for_plot, label=cur_factor)
        plt.title(f'{cur_factor} of countries')
        plt.xlabel('Countries pairs')
        plt.ylabel(cur_factor)
        plt.legend()
        # Save the figure
        plt.savefig(f'results/figure/factors/{cur_factor}_bar_plot.png')  # Save as PNG
        plt.close()

        # trade flow, using absolute values in the summary
        country_pair_trade = pd.read_csv(data_route + "country_info/economy_flow_raw_data/trade_export_flow.csv")
        country_pair_trade = country_pair_trade[["Country Name", "Counterpart Country Name", "Value"]]

        country_pair_trade_dict = {key: country_pair_trade[key].to_list() for key in country_pair_trade.keys()}
        country_pair_trade_value_dict = defaultdict(lambda: defaultdict(float))
        for country_idx in range(len(country_pair_trade["Country Name"])):
            try:
                cur_destination_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Country Name"][country_idx]]
                cur_sending_alpha3 = country_to_alpha3_dict[country_pair_trade_dict["Counterpart Country Name"][country_idx]]

                country_pair_trade_value_dict[cur_destination_alpha3][cur_sending_alpha3] += country_pair_trade_dict["Value"][country_idx]
                country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3] += country_pair_trade_dict["Value"][country_idx]
            except:
                pass

        # plot trade bar plot
        cur_factor = 'trade'
        country_pair_trade_dict_for_plot = []
        for cur_sending_alpha3 in country_pair_trade_value_dict:
            for cur_destination_alpha3 in country_pair_trade_value_dict[cur_sending_alpha3]:
                country_pair_trade_dict_for_plot.append(country_pair_trade_value_dict[cur_sending_alpha3][cur_destination_alpha3])
        plt.figure()      # Create a new figure
        plt.ylim(0,150000000000)
        plt.bar(range(len(country_pair_trade_dict_for_plot)), country_pair_trade_dict_for_plot, label=cur_factor)
        plt.title(f'{cur_factor} of countries')
        plt.xlabel('Countries pairs')
        plt.ylabel(cur_factor)
        plt.legend()
        # Save the figure
        plt.savefig(f'results/figure/factors/{cur_factor}_bar_plot.png')  # Save as PNG
        plt.close()

        # immgration flow
        country_pair_immgration = pd.read_csv(data_route + "country_info/migration/bilateral_migrationmatrix_2018.csv")

        country_pair_immgration_dict = {key: country_pair_immgration[key].to_list() for key in
                                        country_pair_immgration.keys()}
        country_immgration_list = list(country_pair_immgration["Country"])
        country_pair_immgration_dict.pop("Country")
        for cur_country1 in country_pair_immgration_dict:
            for i in range(len(country_pair_immgration_dict[cur_country1])):
                if isinstance(country_pair_immgration_dict[cur_country1][i], str):
                    country_pair_immgration_dict[cur_country1][i] = float(country_pair_immgration_dict[cur_country1][i].replace(",",""))
            country_pair_immgration_dict[cur_country1] = {country_immgration_list[i]:country_pair_immgration_dict[cur_country1][i] for i in range(len(country_immgration_list))}


        country_pair_immgration_value_dict = defaultdict(lambda: defaultdict(float))
        for country1_idx in range(len(country_immgration_list)):
            for country2_idx in range(len(country_immgration_list)):
                try:
                    cur_country1 = country_immgration_list[country1_idx]
                    cur_country2 = country_immgration_list[country2_idx]

                    country_pair_immgration_value_dict[cur_country1][cur_country2] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                    country_pair_immgration_value_dict[cur_country2][cur_country1] += float(
                        country_pair_immgration_dict[cur_country1][cur_country2])
                except:
                    pass

        # # plot immgration bar plot
        # cur_factor = 'immgration'
        # country_pair_immgration_dict_for_plot = []
        # for cur_sending_alpha3 in country_pair_immgration_dict:
        #     for cur_destination_alpha3 in country_pair_immgration_dict[cur_sending_alpha3]:
        #         country_pair_immgration_dict_for_plot.append(country_pair_immgration_dict[cur_sending_alpha3][cur_destination_alpha3])
        # plt.figure()  # Create a new figure
        # plt.bar(range(len(country_pair_immgration_dict_for_plot)), country_pair_immgration_dict_for_plot, label=cur_factor)
        # plt.title(f'{cur_factor} of countries')
        # plt.xlabel('Countries pairs')
        # plt.ylabel(cur_factor)
        # plt.legend()
        # # Save the figure
        # plt.savefig(f'results/figure/factors/{cur_factor}_bar_plot.png')  # Save as PNG
        # plt.close()