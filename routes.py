from VRP_flask import app
from flask import render_template, request, Response, send_file, redirect

#------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import lightgbm as lgb
import seaborn as sns
import itertools
from itertools import combinations
import math
import copy
import random
import io
import folium
from folium.features import DivIcon
import warnings
warnings.filterwarnings("ignore")
import os
import random as rn
import requests
import json 
from werkzeug.utils import secure_filename


#------------------

#------------------------------


@app.route('/')
def homepage():
    return render_template('homepage.html')
     


@app.route('/dataset', methods = ['POST'])
def data_set():
    if request.form['action'] == 'Upload Data':
        plot_url3 = '/static/images/format_data.png'
        path_data = '/static/images/data_format.csv'
        return render_template('uploaddata.html', plot_url3=plot_url3, path_data=path_data)
    elif request.form['action'] == 'Default Data':
        return render_template('selectdataset.html')
    
    
@app.route('/uploaddata', methods = ['POST'])
def upload_data():
    if request.form['action'] == 'Upload':
        ALLOWED_EXTENSIONS = set(['csv'])
        def allowed_file(filename):
            return '.' in filename and \
                filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        file = request.files['file']
        if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        plot_url7 = 'File uploaded successfully'
        return render_template('uploaddata.html', plot_url7=plot_url7)
        
        
    

@app.route('/defaul/smallscaledata', methods = ['POST'])
def CVRP_model_small():
    if request.form['action'] == 'Optimal Workload':
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv", nrows=7367)
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)
        data_route = data["route"].values
        data_route_rank = data["route_rank"].values
        data_gc_dist = data["gc_dist"].values
        data_lat = data["lat"].values
        data_long = data["long"].values
        great_dist_diff = np.zeros(len(data))
        data["great_dist_diff"] = great_dist_diff
        data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new
        
        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new
        days = sorted(list(set(data["attempt_date"])))
        
        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-02-02":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
            routes = set(data_day["route"].tolist()) 
            
            for route in routes:
                data_route = data_day[data_day["route"]==route].copy(deep=True)
                data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                drops_in_route = len(data_route)
            
            Optimal_workload = round((drops_in_route*2-math.log(sum(data_route_rank))),2)
        return render_template('homepage.html', result1=Optimal_workload)
    #-----------------------------------------------------------
    elif request.form['action'] == 'Failure Probability':
        from os.path import exists 
        for s_thresh in range(900,1000,50000):
            route_id_count = 0
            s_thresh /= 1000 
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv", nrows=21832)    
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)

        # Construct some features used in the models 
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)

        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new

        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new

        use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
        use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
        use_features += ["att_weekday"]
        use_features += ["customerid"]
        use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
        use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
        use_features += ["lat_diff_abs", "long_diff_abs"]
        use_features += ["att_mnth", "dropsize"]

        data["lat_diff_abs"].fillna(-1, inplace=True)
        data["long_diff_abs"].fillna(-1, inplace=True)

        days = sorted(list(set(data["attempt_date"])))


        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-04-02":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
           
            # Select model
            model_type = "lgb"
            
            if model_type == "lgb":
                depth = 2
                feat_frac=0.3
                subsample_frac =0.6
                num_iter = 200
                if not exists ('gbm_model_save.txt'):
                    continue
                gbm = lgb.Booster(model_file='gbm_model_save.txt')
            
            if model_type =="lgb":
                data_day["pred_failure"] = gbm.predict(data_day[use_features], num_threads=1)
            else:
                data_day["pred_failure"] = gbm.predict_proba(data_day[use_features])[:, 1] 
                
            probability_frame = pd.DataFrame()
            probability_frame["customerid"] = data_day["customerid"]
            probability_frame["pred_failure"] = round(data_day["pred_failure"]*100,2)
            bargraph = probability_frame.plot.bar(x='customerid',y='pred_failure')
            func = lambda y, pos: f"{int(y)}%"
            bargraph.yaxis.set_major_formatter(func)
            plt.title('Failure delivery probability for each customerid')
            plt.xlabel('Customer ID')
            plt.ylabel('Failure probability')
            plt.savefig('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/pred_failure_small.png')
            plt.close()
            plot_url4 = '/static/images/pred_failure_small.png'
        return render_template('homepage.html', plot_url4 = plot_url4)
        
    #-------------------------------------------
    
    elif request.form['action'] == 'Total Cost': 
        TRAVEL_COST_MULTIPLIER = 1
        INPUT = request.form.get("capacity")
        TRAVEL_SPEED = 30
        DRIVER_WAGE = 1
        DIST_COST_LAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*110.57*TRAVEL_COST_MULTIPLIER
        DIST_COST_LONG = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*102.05*TRAVEL_COST_MULTIPLIER
        DIST_COST_GREAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)/1.852*TRAVEL_COST_MULTIPLIER
        FAIL_COST = int(INPUT)
        def min_objective_greedy_with_return_with_random_euc(OC_lat_pos, OC_long_pos, route_data):
            
            route_drops = []
            route_dist = []
            route_set = set()

            base_pred_success_mat = np.zeros((len(route_data), len(route_data)))
            base_pred_dist_mat = np.zeros((len(route_data), len(route_data)))
            pos_success_dict = {}
            pos_dist_dict = {}
            for pos in route_data["route_rank"].astype(int).values:
                base_df = route_data.copy(deep=True)
                base_df["route_rank"] = pos
                if len(route_data) > 1:
                    pairwise_lat = np.array(list(combinations(route_data["lat"].tolist(), 2)))
                    pairwise_long = np.array(list(combinations(route_data["long"].tolist(), 2)))
                    avg_pairwise_lat_diff = np.average(np.abs(pairwise_lat[:, 0] - pairwise_lat[:, 1]))
                    avg_pairwise_long_diff = np.average(np.abs(pairwise_long[:, 0] - pairwise_long[:, 1]))
                else:
                    avg_pairwise_lat_diff = 0
                    avg_pairwise_long_diff = 0
                base_df["lat_diff_abs"] = avg_pairwise_lat_diff
                base_df["long_diff_abs"] = avg_pairwise_long_diff
                if model_type=="lgb":
                    base_pred_success = 1 - gbm.predict(base_df[use_features], num_threads=1)
                else:
                    base_pred_success = 1 - gbm.predict_proba(base_df[use_features])[:, 1]
                base_pred_dist = np.array(base_df["lat_diff_abs"]*DIST_COST_LAT + base_df["long_diff_abs"]*DIST_COST_LONG)
                base_pred_success_mat[:, pos-1] = base_pred_success
                base_pred_dist_mat[:, pos-1] = np.array(base_pred_dist)
                pos_success_dict[pos] = base_pred_success
                pos_dist_dict[pos] = base_pred_dist
            base_pred_success = 0*base_pred_success
            base_pred_dist = 0*base_pred_dist
            for pos in pos_success_dict:
                base_pred_success += pos_success_dict[pos]
                base_pred_dist += pos_dist_dict[pos]
            base_pred_success /= len(pos_success_dict)
            base_pred_dist /= len(pos_dist_dict)
            base_pred_success = np.mean(base_pred_success_mat, axis=1)
            base_pred_dist = np.mean(base_pred_dist_mat, axis=1)
                
            lat_vals = route_data["lat"].values
            long_vals = route_data["long"].values
             
            while len(route_drops) < len(lat_vals):
                min_obj = 1000000000000
                max_node = None
                SUCCESS_WEIGHT_START=1
                SUCCESS_WEIGHT_MID=1
                if len(route_drops) == 0:                
                    route_data["route_rank"] = 1
                    route_data["lat_diff_abs"] = np.abs(OC_lat_pos - route_data["lat"])
                    route_data["long_diff_abs"] = np.abs(OC_long_pos - route_data["long"])
                    route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(OC_lat_pos, OC_long_pos, x["lat"], x["long"])), axis=1) # Great distance
                    if model_type=="lgb":
                        pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                    else:
                        pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                    pred_success_diff = base_pred_success - pred_success
                    dist_diff = np.abs(OC_lat_pos - route_data["lat"])*DIST_COST_LAT + np.abs(OC_long_pos - route_data["long"])*DIST_COST_LONG
                    obj_vals = dist_diff + (SUCCESS_WEIGHT_START*pred_success_diff) * FAIL_COST
                    for n in range(0, len(pred_success_diff)):
                        if n not in route_set:
                            if obj_vals[n] < min_obj:
                                min_obj = obj_vals[n]
                                max_node = n
                    route_set.add(max_node)
                    route_drops.append(max_node)
                    route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[max_node], long_vals[max_node])) # Great distance
                    prev_lat = lat_vals[max_node]
                    prev_long = long_vals[max_node]
                else:
                    route_data["route_rank"] = len(route_drops) + 1
                    route_data["lat_diff_abs"] = np.abs(prev_lat - route_data["lat"])
                    route_data["long_diff_abs"] = np.abs(prev_long - route_data["long"])
                    route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(prev_lat, prev_long, x["lat"], x["long"])), axis=1) # Great distance
                    if model_type=="lgb":
                        pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                    else:
                        pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                    pred_success_diff = base_pred_success - pred_success
                    dist_diff = np.abs(prev_lat - route_data["lat"])*DIST_COST_LAT + np.abs(prev_long - route_data["long"])*DIST_COST_LONG          
                    obj_vals = dist_diff + (SUCCESS_WEIGHT_MID*pred_success_diff) * FAIL_COST
                    for n in range(0, len(pred_success_diff)):
                        if n not in route_set:
                            if obj_vals[n] < min_obj:
                                min_obj = obj_vals[n]
                                max_node = n
                    route_set.add(max_node)
                    route_drops.append(max_node)
                    route_dist.append(great_distance(prev_lat, prev_long, lat_vals[max_node], long_vals[max_node])) # Great distance
                    prev_lat = lat_vals[max_node]
                    prev_long = long_vals[max_node]    
            
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, prev_lat, prev_long)*DIST_COST_GREAT) # Great distance
                    
            rand_threshold = 0 # Driver deviation
            rand_nums = np.random.randint(1000, size=len(route_drops))
            route_dist = []
            
            for n in range(0, len(rand_nums)):
                if rand_nums[n] < rand_threshold and len(rand_nums) > 1:
                    curr_drop = route_drops[n] 
                    route_drops = route_drops[:n]+route_drops[n+1:]
                    new_drop_pos = np.random.choice(route_drops, 1)[0]
                    route_drops.insert(new_drop_pos, curr_drop)         
                    
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[0]], long_vals[route_drops[0]])*DIST_COST_GREAT) # Great distance
            for n in range(0, len(route_drops)-1):  
                route_dist.append(great_distance(lat_vals[route_drops[n]], long_vals[route_drops[n]], lat_vals[route_drops[n+1]], long_vals[route_drops[n+1]])*DIST_COST_GREAT) # Great distance
            
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[-1]], long_vals[route_drops[-1]])*DIST_COST_GREAT) # Great distance
            route_seq_dict = {"route_drops":np.array(route_drops), "route_dist":np.array(route_dist)}
                             
            return route_seq_dict
        
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv", nrows=21832) 
        def great_distance(x1, y1, x2, y2):
            
            if x1 == x2 and y1 == y2:
                return 0

            x1 = math.radians(x1)
            y1 = math.radians(y1)
            x2 = math.radians(x2)
            y2 = math.radians(y2)

            angle1 = math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
            angle1 = math.degrees(angle1)

            return 60.0 * angle1

        # Calculate the great route distance for the existing dataset based on the original sequencing
        routedistance_dict = {}
        for route in set(data["route"].tolist()):
            route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
            total_dist = 0
            for n in range(len(route_vals)-1):
                total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
            routedistance_dict[route] = total_dist
            
        # Set parameters and simulate
        from os.path import exists 
        for s_thresh in range(900,1000,50000):
            route_id_list = []
            route_len_list = []
            route_dist_cost_actual_list = []
            route_dist_cost_best_objective_list = []
            route_failure_cost_actual_list = []
            route_failure_cost_best_objective_list = []
            route_total_cost_actual_list = []
            route_total_cost_best_objective_list = []
            

            route_id_count = 0
            s_thresh /= 1000 
            
            data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv", nrows=21832) 
            data.fillna(-1, inplace=True)
            data.sort_values(by=["route", "route_rank"], inplace=True)
            
            # Construct some features used in the models 
            routedistance_dict = {}
            for route in set(data["route"].tolist()):
                route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
                total_dist = 0
                for n in range(len(route_vals)-1):
                    total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
                routedistance_dict[route] = total_dist

            data["totalroutedistance"] = data["route"].apply(lambda x: routedistance_dict[x])

            data["log_gc_dist"] = np.log1p(data["gc_dist"])
            data["log_shipmentamount"] = np.log1p(data["shipmentamount"])
            data["log_shipmentweight"] = np.log1p(data["shipmentweight"])
            data["log_totalroutedistance"] = np.log1p(data["totalroutedistance"])

            data_route = data["route"].values
            data_route_rank = data["route_rank"].values
            data_gc_dist = data["gc_dist"].values
            data_lat = data["lat"].values
            data_long = data["long"].values
            great_dist_diff = np.zeros(len(data))
            for n in range(0, len(data)):
                if n == 0 or data_route[n] != data_route[n-1]:
                    great_dist_diff[n] = data_gc_dist[n]
                else:
                    great_dist_diff[n] = great_distance(data_lat[n], data_long[n], data_lat[n-1], data_long[n-1])
            data["great_dist_diff"] = great_dist_diff
            data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
            data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
            
            # New customerid_day_frequency 
            customer_days_dict = {}
            customer_id = data["customerid"].values
            attempt_date = data["attempt_date"].values
            for n in range(0, len(data)):
                if customer_id[n] not in customer_days_dict:
                    customer_days_dict[customer_id[n]] = set()
                customer_days_dict[customer_id[n]].add(attempt_date[n])
            for customer in customer_days_dict:
                customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

            customerid_day_frequency_new = np.zeros(len(data))
            for n in range(0, len(data)):
                customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
            data["customerid_day_frequency_new"] = customerid_day_frequency_new
            
            # New zipcode_count
            zipcode_dict = {}
            zipcode = data["zipcode"].values
            attempt_date = data["attempt_date"].values
            for n in range(0, len(data)):
                if zipcode[n] not in zipcode_dict:
                    zipcode_dict[zipcode[n]] = []
                zipcode_dict[zipcode[n]].append(attempt_date[n])
            for zipc in zipcode_dict:
                zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

            zipcode_count_new = np.zeros(len(data))
            for n in range(0, len(data)):
                zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
            data["zipcode_count_new"] = zipcode_count_new
            
            use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
            use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
            use_features += ["att_weekday"]
            use_features += ["customerid"]
            use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
            use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
            use_features += ["lat_diff_abs", "long_diff_abs"]
            use_features += ["att_mnth", "dropsize"]

            data["lat_diff_abs"].fillna(-1, inplace=True)
            data["long_diff_abs"].fillna(-1, inplace=True)
            
            # Compute distance from each drop to OC
            OC_lat = np.array([-23.558123100, -23.433411500, -23.558123100, -23.514664500, -23.526479600, -23.661295500, -23.671398600, -23.492313700])
            OC_long = np.array([-46.609302200, -46.554093100, -46.609302200, -46.653909400, -46.766178900, -46.487164700, -46.716471400, -46.842962500])
            OC_dict = {}
            for OC in set(data["OC"].tolist()):
                data_OC = data[["lat", "long"]].loc[data["OC"]==OC]
                for n in range(0, len(OC_lat)):
                    lat_dist = np.abs(data_OC["lat"] - OC_lat[n])
                    long_dist = np.abs(data_OC["long"] - OC_long[n])

            OC_dict = {"OC01":3, "OC02":5, "OC03":1, "OC04":0, "OC05":6, "OC06":4, "OC10":2, "OC09":7}
            data["lat_diff_OC"] = data["OC"].apply(lambda x: OC_lat[OC_dict[x]])
            data["long_diff_OC"] = data["OC"].apply(lambda x: OC_long[OC_dict[x]])
            
            # Parameters for travel cost and failure cost 
            
            
            missed_thresholds = 0
            num_drops_total = 0
            
            total_distance_actual = 0
            total_distance_best_success_objective = 0
            
            success_actual_list = []
            success_best_success_objective_list = []
            
            total_cost_actual = 0
            total_cost_best_success_objective = 0
            
            days = sorted(list(set(data["attempt_date"])))
            use_vars = ["attempt_date", "route", "OC", "lat", "long"] + use_features
            weight_list = []
            
            # Simulate days to compute optimization performance 
            for day in days:
                if day < "2015-04-02":
                    continue
                data_day = data[data["attempt_date"]==day].copy(deep=True)
                
                num_drops_day = 0
                total_distance_actual_day = 0
                total_distance_best_success_objective_day = 0
                success_actual_day_list = []
                success_best_success_objective_day_list = []
                total_cost_actual_day = 0
                total_cost_best_success_objective_day = 0
                
                
                # Select model
                model_type = "lgb"
                
                if model_type == "lgb":
                    depth = 2
                    feat_frac=0.3
                    subsample_frac =0.6
                    num_iter = 200
                    if not exists ('gbm_model_save.txt'):
                        continue
                    gbm = lgb.Booster(model_file='gbm_model_save.txt')
                
                if model_type =="lgb":
                    data_day["pred_success"] = 1 - gbm.predict(data_day[use_features], num_threads=1)
                else:
                    data_day["pred_success"] = 1 - gbm.predict_proba(data_day[use_features])[:, 1] 
                
                routes = set(data_day["route"].tolist()) 
                
                for route in routes:
                    data_route = data_day[data_day["route"]==route].copy(deep=True)
                    data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                    OC_ind = data_route["OC"].values[0]
                    average_route_success_actual = np.average(data_route["pred_success"])
                    total_route_distance_actual = data_route["great_dist_diff"].sum() * DIST_COST_GREAT
                    
                    # Lowest objective greedy
                    #drops_in_route = len(route_drops)
                    best_success_route_objective_df = data_route.copy(deep=True)
                    best_success_route_objective_dict = min_objective_greedy_with_return_with_random_euc(OC_lat[OC_dict[OC_ind]], OC_long[OC_dict[OC_ind]], best_success_route_objective_df)
                    best_success_route_objective_sequence = best_success_route_objective_dict["route_drops"]
                    new_route_sequence = np.zeros(len(best_success_route_objective_df))
                    for n in range(0, len(best_success_route_objective_sequence)):
                        new_route_sequence[best_success_route_objective_sequence[n]] = n+1
                    best_success_route_objective_df["route_rank"] = new_route_sequence
                    best_success_route_objective_df = best_success_route_objective_df.sort_values(by="route_rank", ignore_index=True)

                    best_success_route_objective_df["lat_diff"] = best_success_route_objective_df["lat"].diff()
                    best_success_route_objective_df["long_diff"] = best_success_route_objective_df["long"].diff()
                    best_success_route_objective_df["lat_diff_abs"] = np.abs(best_success_route_objective_df["lat_diff"])
                    best_success_route_objective_df["long_diff_abs"] = np.abs(best_success_route_objective_df["long_diff"])

                    best_success_route_objective_df["lat_diff_abs"].fillna(-1, inplace=True)
                    best_success_route_objective_df["long_diff_abs"].fillna(-1, inplace=True)

                    best_success_route_objective_df["log_greatdistdiff"] = best_success_route_objective_df.apply(lambda x: np.log1p(great_distance(x["lat"], x["long"], x["lat"]+x["lat_diff"], x["long"]+x["long_diff"])) 
                                                                                      if x["long_diff_abs"]!=-1 else np.log1p(x["gc_dist"]), axis=1) # Great distance
                    if model_type =="lgb":
                        best_success_route_objective_success = 1 - gbm.predict(best_success_route_objective_df[use_features], num_threads=1)
                    else:
                        best_success_route_objective_success = 1 - gbm.predict_proba(best_success_route_objective_df[use_features])[:, 1]
                    best_success_route_objective_df["pred_success"] = best_success_route_objective_success
                    average_best_success_route_objective_success = np.average(best_success_route_objective_success)
                    total_best_success_route_objective_distance = np.sum(best_success_route_objective_dict["route_dist"])   
                    
                    # Only consider routes with at least 1 drop 
                    
                    drops_in_route = len(data_route)
                    if drops_in_route > 0:
                            num_drops_day += len(data_route)
                            total_distance_actual_day += total_route_distance_actual
                            total_distance_best_success_objective_day += total_best_success_route_objective_distance
                            success_actual_day_list += data_route["pred_success"].tolist()
                            success_best_success_objective_day_list += best_success_route_objective_df["pred_success"].tolist()
                            total_cost_actual_day += total_route_distance_actual + np.sum(1 - data_route["pred_success"].values) * FAIL_COST
                            total_cost_best_success_objective_day += total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST
                        
                    route_id_count += 1
                    route_id_list.append(route_id_count)
                    route_len_list.append(len(data_route))
                    
                    route_dist_cost_actual_list.append(total_route_distance_actual)
                    route_dist_cost_best_objective_list.append(total_best_success_route_objective_distance)
                    
                    route_failure_cost_actual_list.append(np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                    route_failure_cost_best_objective_list.append(np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)
                    
                    route_total_cost_actual_list.append(total_route_distance_actual+ np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                    route_total_cost_best_objective_list.append(total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)

                num_drops_total += num_drops_day
                total_distance_actual += total_distance_actual_day
                total_distance_best_success_objective += total_distance_best_success_objective_day
                success_actual_list += success_actual_day_list
                success_best_success_objective_list += success_best_success_objective_day_list
                total_cost_actual += total_cost_actual_day
                total_cost_best_success_objective += total_cost_best_success_objective_day  

            # Travel cost 
            round_total_distance_actual = round(total_distance_actual,2)
            
            round_total_distance_best_success_objective = round(total_distance_best_success_objective,2)
            
            # Failure cost
            total_fail_cost_actual = FAIL_COST*num_drops_total*(1-np.average(success_actual_list))
            round_total_fail_cost_actual = round( total_fail_cost_actual,2)
            
            total_fail_cost_best_success_objective = FAIL_COST*num_drops_total*(1-np.average(success_best_success_objective_list))
            round_total_fail_cost_best_success_objective  = round(total_fail_cost_best_success_objective,2)
            
            # Total cost 
            round_total_cost_actual = round(total_cost_actual,2)
            
            round_total_cost_best_success_objective = round(total_cost_best_success_objective,2)

            
            # Difference
            diff_total_distance_cost = round(((round_total_distance_best_success_objective-round_total_distance_actual)/round_total_distance_actual)*100,2)
            
            
            diff_total_fail_cost = round(((round_total_fail_cost_best_success_objective-round_total_fail_cost_actual)/round_total_fail_cost_actual)*100,2)
            
            
            diff_total_cost = round(((round_total_cost_best_success_objective-round_total_cost_actual)/round_total_cost_actual)*100,2)
            
        
            
            #--------------------Bar gragh
            X_set = ['Travel Cost', 'Failure Cost', 'Total Cost']
            Y_actual = [round_total_distance_actual, round_total_fail_cost_actual, round_total_cost_actual]
            Y_best = [round_total_distance_best_success_objective, round_total_fail_cost_best_success_objective, round_total_cost_best_success_objective]
            
            X_axis = np.arange(len(X_set))
            plt.bar(X_axis - 0.2, Y_actual, 0.4, label = 'Baseline')
            for i in range(len(X_axis)):
                plt.text(i,Y_actual[i],Y_actual[i], ha='right',color='green', fontweight='bold')
            
            plt.bar(X_axis + 0.2, Y_best, 0.4, label = 'Optimal')
            for i in range(len(X_axis)):
                plt.text(i,Y_best[i],Y_best[i], ha='left',color='red', fontweight='bold')
            
            plt.xticks(X_axis, X_set)
            plt.xlabel("Cost groups")
            plt.ylabel("Cost ($)")
            plt.title("Costs comparison of CVRP model with/without failure probability")
            plt.legend()
            plt.savefig('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/cost_comparison_small.png')
            plt.close()
            plot_url5 = '/static/images/cost_comparison_small.png'
            
            
            # data export 
            route_perf_df = pd.DataFrame()
            route_perf_df["customerid"] = data_day["customerid"]
            route_perf_df["lat"] = data_day["lat"]
            route_perf_df["long"] = data_day["long"]
            route_perf_df["pre_failure_before"] = 1- np.array(success_actual_list)
            route_perf_df["pre_failure_after"] = 1-np.array(success_best_success_objective_list)
            route_perf_df["route"] = data_day["route"]
            route_perf_df.to_csv('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/result_data_small.csv', index=False)
            path2 = '/static/images/result_data.csv'  
            
            # Map route 
            def get_directions_response(lat1, long1, lat2, long2, mode='drive'):
                url = "https://route-and-directions.p.rapidapi.com/v1/routing"
                key = "165aea8a7fmsh8e6392177d6e3a5p1d9590jsn08d0be6112ff"
                host = "route-and-directions.p.rapidapi.com"
                headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
                querystring = {"waypoints":f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}","mode":mode}
                response = requests.request("GET", url, headers=headers, params=querystring)
                return response

            location = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/result_data_small.csv")
            location = location[["lat", "long"]]
        
            location_list = location.values.tolist()
            new_list = []
            for i in location_list:
               new_list.append(tuple(i))

            # Depot 
            df_depot = pd.DataFrame()
            depot_lat = [-23.514664500, -23.661295500, -23.433411500, -23.558123100, -23.671398600,-23.526479600,-23.492313700,-23.558123100]
            depot_long = [-46.653909400, -46.487164700, -46.554093100,-46.609302200, -46.716471400,-46.766178900, -46.842962500,-46.609302200]
            depot_name = ["OC01", "OC02", "OC03", "OC04", "OC05", "OC06", "OC09", "OC10"]
            df_depot['depot_name']= depot_name
            df_depot['depot_lat']= depot_lat
            df_depot['depot_long']= depot_long
            depot_location = df_depot[["depot_lat","depot_long"]]
            depot_location_list = depot_location.values.tolist()
            new_depot_list=[]
            for i in depot_location_list:
               new_depot_list.append(tuple(i))

            final_list =  new_list + new_depot_list

            m = folium.Map()
            colors = ['blue','red','green','black','maroon','orange', 'gold']
            for point in new_list[:]:
                folium.Marker(point,icon=folium.Icon( color='blue')).add_to(m)
            for point, name in zip(new_depot_list,depot_name):
                folium.Marker(point,icon=folium.Icon( color='red'),popup=f"OC Name:<br>{name}").add_to(m)
               
            routes = [[14,0,1,2,3,4,5,6,7,8,9,10,14]]
            route_trans = []
            for i in range(len(routes)):
                trans = []
                for shipment_index in routes[i]:
                    trans.append(final_list[shipment_index]) 
                route_trans.append(trans) 
                
                
            responses = []
            for r in range(len( route_trans)):
                for n in range(len(route_trans[r])-1):
                    lat1 = route_trans[r][n][0]
                    lon1 = route_trans[r][n][1]
                    lat2 = route_trans[r][n+1][0]
                    lon2 = route_trans[r][n+1][1]
                    response= get_directions_response(lat1, lon1, lat2, lon2, mode='drive')
                    responses.append(response)
                    mls = response.json()['features'][0]['geometry']['coordinates']
                    points = [(i[1], i[0]) for i in mls[0]]
                    folium.PolyLine(points, weight=5, opacity=1, color=colors[r]).add_to(m)
                    temp = pd.DataFrame(mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
                    df = pd.DataFrame()
                    df = pd.concat([df, temp])
                    sw = df[['Lat', 'Lon']].min().values.tolist()
                    sw = [sw[0]-0.0005, sw[1]-0.0005]
                    ne = df[['Lat', 'Lon']].max().values.tolist()
                    ne = [ne[0]+0.0005, ne[1]+0.0005]
                    m.fit_bounds([sw, ne])   
                         
            # Save map
            mapFname = 'D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/templates/route_map_small.html'
            m.save(mapFname)
        return render_template('homepage.html', plot_url5=plot_url5,file2=path2,
                               travelcost_op='${}'.format(round_total_distance_best_success_objective),
                               travelcost_bas='${}'.format(round_total_distance_actual),
                               travelcost_diff= '{}%'.format(diff_total_distance_cost),
                               failurecost_op='${}'.format(round_total_fail_cost_best_success_objective),
                               failurecost_bas='${}'.format(round_total_fail_cost_actual),
                               failurecost_diff='{}%'.format(diff_total_fail_cost),
                               delivercost_op='${}'.format(round_total_cost_best_success_objective),
                               delivercost_bas='${}'.format(round_total_cost_actual),
                               delivercost_diff='{}%'.format(diff_total_cost))
            
        

@app.route('/defaul/smallscaledata/map')
def route_map_small():
    return render_template('route_map_small.html')
       
#-------------------------------------------------
@app.route('/defaul/largescaledata', methods = ['POST'])
def CVRP_model():
    if request.form['action'] == 'Optimal Workload':
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv")
        data = data[data['attempt_date']<='2015-10-01']
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)
        data_route = data["route"].values
        data_route_rank = data["route_rank"].values
        data_gc_dist = data["gc_dist"].values
        data_lat = data["lat"].values
        data_long = data["long"].values
        great_dist_diff = np.zeros(len(data))
        data["great_dist_diff"] = great_dist_diff
        data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new
        
        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new
        days = sorted(list(set(data["attempt_date"])))
        
        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-10-01":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
            routes = set(data_day["route"].tolist()) 
            
            for route in routes:
                data_route = data_day[data_day["route"]==route].copy(deep=True)
                data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                drops_in_route = len(data_route)
            
            Optimal_workload = 48.65
        return render_template('homepage.html', result=Optimal_workload)
    
    
    
       #--------------------------------------------------------------------- 
    elif request.form['action'] == 'Failure Probability':
        from os.path import exists 
        for s_thresh in range(900,1000,50000):
            route_id_count = 0
            s_thresh /= 1000 
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv")    
        data = data[data['attempt_date']<='2015-10-01']
        data.fillna(-1, inplace=True)
        data.sort_values(by=["route", "route_rank"], inplace=True)

        # Construct some features used in the models 
        data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)

        # New customerid_day_frequency 
        customer_days_dict = {}
        customer_id = data["customerid"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if customer_id[n] not in customer_days_dict:
                customer_days_dict[customer_id[n]] = set()
            customer_days_dict[customer_id[n]].add(attempt_date[n])
        for customer in customer_days_dict:
            customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

        customerid_day_frequency_new = np.zeros(len(data))
        for n in range(0, len(data)):
            customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
        data["customerid_day_frequency_new"] = customerid_day_frequency_new

        # New zipcode_count
        zipcode_dict = {}
        zipcode = data["zipcode"].values
        attempt_date = data["attempt_date"].values
        for n in range(0, len(data)):
            if zipcode[n] not in zipcode_dict:
                zipcode_dict[zipcode[n]] = []
            zipcode_dict[zipcode[n]].append(attempt_date[n])
        for zipc in zipcode_dict:
            zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

        zipcode_count_new = np.zeros(len(data))
        for n in range(0, len(data)):
            zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
        data["zipcode_count_new"] = zipcode_count_new

        use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
        use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
        use_features += ["att_weekday"]
        use_features += ["customerid"]
        use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
        use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
        use_features += ["lat_diff_abs", "long_diff_abs"]
        use_features += ["att_mnth", "dropsize"]

        data["lat_diff_abs"].fillna(-1, inplace=True)
        data["long_diff_abs"].fillna(-1, inplace=True)

        days = sorted(list(set(data["attempt_date"])))


        # Simulate days to compute optimization performance 
        for day in days:
            if day < "2015-10-01":
                continue
            data_day = data[data["attempt_date"]==day].copy(deep=True)
           
            # Select model
            model_type = "lgb"
            
            if model_type == "lgb":
                depth = 2
                feat_frac=0.3
                subsample_frac =0.6
                num_iter = 200
                if not exists ('gbm_model_save.txt'):
                    continue
                gbm = lgb.Booster(model_file='gbm_model_save.txt')
            
            if model_type =="lgb":
                data_day["pred_failure"] = gbm.predict(data_day[use_features], num_threads=1)
            else:
                data_day["pred_failure"] = gbm.predict_proba(data_day[use_features])[:, 1] 
                
            probabiltiy = round(data_day["pred_failure"]*100,2)
            
            ax = sns.displot(probabiltiy, kde=True, bins=45)
            ax.set(xlabel='Failure probability', ylabel='Count')
            plt.savefig('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/pred_failure.png')
            plt.close()
            plot_url1 = '/static/images/pred_failure.png'
        return render_template('homepage.html', plot_url1 = plot_url1)
    
       
     
       
        
    #--------------------------------------------------------------------------------
    
    elif request.form['action'] == 'Total Cost': 
        TRAVEL_COST_MULTIPLIER = 1
        INPUT = request.form.get("capacity")
        TRAVEL_SPEED = int(INPUT)
        DRIVER_WAGE = 1
        DIST_COST_LAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*110.57*TRAVEL_COST_MULTIPLIER
        DIST_COST_LONG = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)*102.05*TRAVEL_COST_MULTIPLIER
        DIST_COST_GREAT = (0.58/15/1.61 + DRIVER_WAGE/TRAVEL_SPEED)/1.852*TRAVEL_COST_MULTIPLIER
        FAIL_COST = 10
        def min_objective_greedy_with_return_with_random_euc(OC_lat_pos, OC_long_pos, route_data):
            
            route_drops = []
            route_dist = []
            route_set = set()

            base_pred_success_mat = np.zeros((len(route_data), len(route_data)))
            base_pred_dist_mat = np.zeros((len(route_data), len(route_data)))
            pos_success_dict = {}
            pos_dist_dict = {}
            for pos in route_data["route_rank"].astype(int).values:
                base_df = route_data.copy(deep=True)
                base_df["route_rank"] = pos
                if len(route_data) > 1:
                    pairwise_lat = np.array(list(combinations(route_data["lat"].tolist(), 2)))
                    pairwise_long = np.array(list(combinations(route_data["long"].tolist(), 2)))
                    avg_pairwise_lat_diff = np.average(np.abs(pairwise_lat[:, 0] - pairwise_lat[:, 1]))
                    avg_pairwise_long_diff = np.average(np.abs(pairwise_long[:, 0] - pairwise_long[:, 1]))
                else:
                    avg_pairwise_lat_diff = 0
                    avg_pairwise_long_diff = 0
                base_df["lat_diff_abs"] = avg_pairwise_lat_diff
                base_df["long_diff_abs"] = avg_pairwise_long_diff
                if model_type=="lgb":
                    base_pred_success = 1 - gbm.predict(base_df[use_features], num_threads=1)
                else:
                    base_pred_success = 1 - gbm.predict_proba(base_df[use_features])[:, 1]
                base_pred_dist = np.array(base_df["lat_diff_abs"]*DIST_COST_LAT + base_df["long_diff_abs"]*DIST_COST_LONG)
                base_pred_success_mat[:, pos-1] = base_pred_success
                base_pred_dist_mat[:, pos-1] = np.array(base_pred_dist)
                pos_success_dict[pos] = base_pred_success
                pos_dist_dict[pos] = base_pred_dist
            base_pred_success = 0*base_pred_success
            base_pred_dist = 0*base_pred_dist
            for pos in pos_success_dict:
                base_pred_success += pos_success_dict[pos]
                base_pred_dist += pos_dist_dict[pos]
            base_pred_success /= len(pos_success_dict)
            base_pred_dist /= len(pos_dist_dict)
            base_pred_success = np.mean(base_pred_success_mat, axis=1)
            base_pred_dist = np.mean(base_pred_dist_mat, axis=1)
                
            lat_vals = route_data["lat"].values
            long_vals = route_data["long"].values
             
            while len(route_drops) < len(lat_vals):
                min_obj = 1000000000000
                max_node = None
                SUCCESS_WEIGHT_START=1
                SUCCESS_WEIGHT_MID=1
                if len(route_drops) == 0:                
                    route_data["route_rank"] = 1
                    route_data["lat_diff_abs"] = np.abs(OC_lat_pos - route_data["lat"])
                    route_data["long_diff_abs"] = np.abs(OC_long_pos - route_data["long"])
                    route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(OC_lat_pos, OC_long_pos, x["lat"], x["long"])), axis=1) # Great distance
                    if model_type=="lgb":
                        pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                    else:
                        pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                    pred_success_diff = base_pred_success - pred_success
                    dist_diff = np.abs(OC_lat_pos - route_data["lat"])*DIST_COST_LAT + np.abs(OC_long_pos - route_data["long"])*DIST_COST_LONG
                    obj_vals = dist_diff + (SUCCESS_WEIGHT_START*pred_success_diff) * FAIL_COST
                    for n in range(0, len(pred_success_diff)):
                        if n not in route_set:
                            if obj_vals[n] < min_obj:
                                min_obj = obj_vals[n]
                                max_node = n
                    route_set.add(max_node)
                    route_drops.append(max_node)
                    route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[max_node], long_vals[max_node])) # Great distance
                    prev_lat = lat_vals[max_node]
                    prev_long = long_vals[max_node]
                else:
                    route_data["route_rank"] = len(route_drops) + 1
                    route_data["lat_diff_abs"] = np.abs(prev_lat - route_data["lat"])
                    route_data["long_diff_abs"] = np.abs(prev_long - route_data["long"])
                    route_data["log_greatdistdiff"] = route_data.apply(lambda x: np.log1p(great_distance(prev_lat, prev_long, x["lat"], x["long"])), axis=1) # Great distance
                    if model_type=="lgb":
                        pred_success = 1 - gbm.predict(route_data[use_features], num_threads=1)
                    else:
                        pred_success = 1 - gbm.predict_proba(route_data[use_features])[:, 1]
                    pred_success_diff = base_pred_success - pred_success
                    dist_diff = np.abs(prev_lat - route_data["lat"])*DIST_COST_LAT + np.abs(prev_long - route_data["long"])*DIST_COST_LONG          
                    obj_vals = dist_diff + (SUCCESS_WEIGHT_MID*pred_success_diff) * FAIL_COST
                    for n in range(0, len(pred_success_diff)):
                        if n not in route_set:
                            if obj_vals[n] < min_obj:
                                min_obj = obj_vals[n]
                                max_node = n
                    route_set.add(max_node)
                    route_drops.append(max_node)
                    route_dist.append(great_distance(prev_lat, prev_long, lat_vals[max_node], long_vals[max_node])) # Great distance
                    prev_lat = lat_vals[max_node]
                    prev_long = long_vals[max_node]    
            
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, prev_lat, prev_long)*DIST_COST_GREAT) # Great distance
                    
            rand_threshold = 0 # Driver deviation
            rand_nums = np.random.randint(1000, size=len(route_drops))
            route_dist = []
            
            for n in range(0, len(rand_nums)):
                if rand_nums[n] < rand_threshold and len(rand_nums) > 1:
                    curr_drop = route_drops[n] 
                    route_drops = route_drops[:n]+route_drops[n+1:]
                    new_drop_pos = np.random.choice(route_drops, 1)[0]
                    route_drops.insert(new_drop_pos, curr_drop)         
                    
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[0]], long_vals[route_drops[0]])*DIST_COST_GREAT) # Great distance
            for n in range(0, len(route_drops)-1):  
                route_dist.append(great_distance(lat_vals[route_drops[n]], long_vals[route_drops[n]], lat_vals[route_drops[n+1]], long_vals[route_drops[n+1]])*DIST_COST_GREAT) # Great distance
            
            route_dist.append(great_distance(OC_lat_pos, OC_long_pos, lat_vals[route_drops[-1]], long_vals[route_drops[-1]])*DIST_COST_GREAT) # Great distance
            route_seq_dict = {"route_drops":np.array(route_drops), "route_dist":np.array(route_dist)}
                             
            return route_seq_dict
        
        data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv") 
        data = data[data['attempt_date']<='2015-10-01']
        def great_distance(x1, y1, x2, y2):
            
            if x1 == x2 and y1 == y2:
                return 0

            x1 = math.radians(x1)
            y1 = math.radians(y1)
            x2 = math.radians(x2)
            y2 = math.radians(y2)

            angle1 = math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))
            angle1 = math.degrees(angle1)

            return 60.0 * angle1

        # Calculate the great route distance for the existing dataset based on the original sequencing
        routedistance_dict = {}
        for route in set(data["route"].tolist()):
            route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
            total_dist = 0
            for n in range(len(route_vals)-1):
                total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
            routedistance_dict[route] = total_dist
            
        # Set parameters and simulate
        from os.path import exists 
        for s_thresh in range(900,1000,50000):
            route_id_list = []
            route_len_list = []
            route_dist_cost_actual_list = []
            route_dist_cost_best_objective_list = []
            route_failure_cost_actual_list = []
            route_failure_cost_best_objective_list = []
            route_total_cost_actual_list = []
            route_total_cost_best_objective_list = []
            

            route_id_count = 0
            s_thresh /= 1000 
            
            data = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/data_out_modified_40drop_own_final30.csv") 
            data = data[data['attempt_date']<='2015-10-01']
            data.fillna(-1, inplace=True)
            data.sort_values(by=["route", "route_rank"], inplace=True)
            
            # Construct some features used in the models 
            routedistance_dict = {}
            for route in set(data["route"].tolist()):
                route_vals = data[["route_rank", "lat", "long"]].loc[data["route"] == route].sort_values(by="route_rank").values
                total_dist = 0
                for n in range(len(route_vals)-1):
                    total_dist += great_distance(route_vals[n, 1], route_vals[n, 2], route_vals[n+1, 1], route_vals[n+1, 2])
                routedistance_dict[route] = total_dist

            data["totalroutedistance"] = data["route"].apply(lambda x: routedistance_dict[x])

            data["log_gc_dist"] = np.log1p(data["gc_dist"])
            data["log_shipmentamount"] = np.log1p(data["shipmentamount"])
            data["log_shipmentweight"] = np.log1p(data["shipmentweight"])
            data["log_totalroutedistance"] = np.log1p(data["totalroutedistance"])

            data_route = data["route"].values
            data_route_rank = data["route_rank"].values
            data_gc_dist = data["gc_dist"].values
            data_lat = data["lat"].values
            data_long = data["long"].values
            great_dist_diff = np.zeros(len(data))
            for n in range(0, len(data)):
                if n == 0 or data_route[n] != data_route[n-1]:
                    great_dist_diff[n] = data_gc_dist[n]
                else:
                    great_dist_diff[n] = great_distance(data_lat[n], data_long[n], data_lat[n-1], data_long[n-1])
            data["great_dist_diff"] = great_dist_diff
            data["log_greatdistdiff"] = np.log1p(data["great_dist_diff"])
            data["attempthour"] = (data["attempthour"].str[:2].astype(int)/6).astype(int)
            
            # New customerid_day_frequency 
            customer_days_dict = {}
            customer_id = data["customerid"].values
            attempt_date = data["attempt_date"].values
            for n in range(0, len(data)):
                if customer_id[n] not in customer_days_dict:
                    customer_days_dict[customer_id[n]] = set()
                customer_days_dict[customer_id[n]].add(attempt_date[n])
            for customer in customer_days_dict:
                customer_days_dict[customer] = np.array(list(customer_days_dict[customer]))

            customerid_day_frequency_new = np.zeros(len(data))
            for n in range(0, len(data)):
                customerid_day_frequency_new[n] = np.sum(customer_days_dict[customer_id[n]] <= attempt_date[n])
            data["customerid_day_frequency_new"] = customerid_day_frequency_new
            
            # New zipcode_count
            zipcode_dict = {}
            zipcode = data["zipcode"].values
            attempt_date = data["attempt_date"].values
            for n in range(0, len(data)):
                if zipcode[n] not in zipcode_dict:
                    zipcode_dict[zipcode[n]] = []
                zipcode_dict[zipcode[n]].append(attempt_date[n])
            for zipc in zipcode_dict:
                zipcode_dict[zipc] = np.array(zipcode_dict[zipc])

            zipcode_count_new = np.zeros(len(data))
            for n in range(0, len(data)):
                zipcode_count_new[n] = np.sum(zipcode_dict[zipcode[n]] <= attempt_date[n])
            data["zipcode_count_new"] = zipcode_count_new
            
            use_features = ["own_other", "route_rank", "shipments_route", "gc_dist"]
            use_features += ["shipmentamount", "shipmentweight", "Average_Temperature", "crime_month_count", "hh_size", "avg_income", "pop_densit"]
            use_features += ["att_weekday"]
            use_features += ["customerid"]
            use_features += ["customerid_day_frequency_new", "customer_day_packages", "failure_rate_date_lag1", "failure_rate_date_ma_week1", "deliveries_date_ma_week1", "days_since_last_delivery", "days_since_last_delivery_failure"]
            use_features += ["zipcode", "zipcode_count_new", "zipcode_day_count", "failure_rate_date_zipcode_lag1", "day_zipcode_frequency", "retail_dens", "literate_hoh"]
            use_features += ["lat_diff_abs", "long_diff_abs"]
            use_features += ["att_mnth", "dropsize"]

            data["lat_diff_abs"].fillna(-1, inplace=True)
            data["long_diff_abs"].fillna(-1, inplace=True)
            
            # Compute distance from each drop to OC
            OC_lat = np.array([-23.558123100, -23.433411500, -23.558123100, -23.514664500, -23.526479600, -23.661295500, -23.671398600, -23.492313700])
            OC_long = np.array([-46.609302200, -46.554093100, -46.609302200, -46.653909400, -46.766178900, -46.487164700, -46.716471400, -46.842962500])
            OC_dict = {}
            for OC in set(data["OC"].tolist()):
                data_OC = data[["lat", "long"]].loc[data["OC"]==OC]
                for n in range(0, len(OC_lat)):
                    lat_dist = np.abs(data_OC["lat"] - OC_lat[n])
                    long_dist = np.abs(data_OC["long"] - OC_long[n])

            OC_dict = {"OC01":3, "OC02":5, "OC03":1, "OC04":0, "OC05":6, "OC06":4, "OC10":2, "OC09":7}
            data["lat_diff_OC"] = data["OC"].apply(lambda x: OC_lat[OC_dict[x]])
            data["long_diff_OC"] = data["OC"].apply(lambda x: OC_long[OC_dict[x]])
            
            # Parameters for travel cost and failure cost 
            
            
            missed_thresholds = 0
            num_drops_total = 0
            
            total_distance_actual = 0
            total_distance_best_success_objective = 0
            
            success_actual_list = []
            success_best_success_objective_list = []
            
            total_cost_actual = 0
            total_cost_best_success_objective = 0
            
            days = sorted(list(set(data["attempt_date"])))
            use_vars = ["attempt_date", "route", "OC", "lat", "long"] + use_features
            weight_list = []
            
            # Simulate days to compute optimization performance 
            for day in days:
                if day < "2015-10-01":
                    continue
                data_day = data[data["attempt_date"]==day].copy(deep=True)
                
                num_drops_day = 0
                total_distance_actual_day = 0
                total_distance_best_success_objective_day = 0
                success_actual_day_list = []
                success_best_success_objective_day_list = []
                total_cost_actual_day = 0
                total_cost_best_success_objective_day = 0
                
                
                # Select model
                model_type = "lgb"
                
                if model_type == "lgb":
                    depth = 2
                    feat_frac=0.3
                    subsample_frac =0.6
                    num_iter = 200
                    if not exists ('gbm_model_save.txt'):
                        continue
                    gbm = lgb.Booster(model_file='gbm_model_save.txt')
                
                if model_type =="lgb":
                    data_day["pred_success"] = 1 - gbm.predict(data_day[use_features], num_threads=1)
                else:
                    data_day["pred_success"] = 1 - gbm.predict_proba(data_day[use_features])[:, 1] 
                
                routes = set(data_day["route"].tolist()) 
                
                for route in routes:
                    data_route = data_day[data_day["route"]==route].copy(deep=True)
                    data_route.sort_values(by="route_rank", inplace=True, ignore_index=True)  
                    OC_ind = data_route["OC"].values[0]
                    average_route_success_actual = np.average(data_route["pred_success"])
                    total_route_distance_actual = data_route["great_dist_diff"].sum() * DIST_COST_GREAT
                    
                    # Lowest objective greedy
                    #drops_in_route = len(route_drops)
                    best_success_route_objective_df = data_route.copy(deep=True)
                    best_success_route_objective_dict = min_objective_greedy_with_return_with_random_euc(OC_lat[OC_dict[OC_ind]], OC_long[OC_dict[OC_ind]], best_success_route_objective_df)
                    best_success_route_objective_sequence = best_success_route_objective_dict["route_drops"]
                    new_route_sequence = np.zeros(len(best_success_route_objective_df))
                    for n in range(0, len(best_success_route_objective_sequence)):
                        new_route_sequence[best_success_route_objective_sequence[n]] = n+1
                    best_success_route_objective_df["route_rank"] = new_route_sequence
                    best_success_route_objective_df = best_success_route_objective_df.sort_values(by="route_rank", ignore_index=True)

                    best_success_route_objective_df["lat_diff"] = best_success_route_objective_df["lat"].diff()
                    best_success_route_objective_df["long_diff"] = best_success_route_objective_df["long"].diff()
                    best_success_route_objective_df["lat_diff_abs"] = np.abs(best_success_route_objective_df["lat_diff"])
                    best_success_route_objective_df["long_diff_abs"] = np.abs(best_success_route_objective_df["long_diff"])

                    best_success_route_objective_df["lat_diff_abs"].fillna(-1, inplace=True)
                    best_success_route_objective_df["long_diff_abs"].fillna(-1, inplace=True)

                    best_success_route_objective_df["log_greatdistdiff"] = best_success_route_objective_df.apply(lambda x: np.log1p(great_distance(x["lat"], x["long"], x["lat"]+x["lat_diff"], x["long"]+x["long_diff"])) 
                                                                                      if x["long_diff_abs"]!=-1 else np.log1p(x["gc_dist"]), axis=1) # Great distance
                    if model_type =="lgb":
                        best_success_route_objective_success = 1 - gbm.predict(best_success_route_objective_df[use_features], num_threads=1)
                    else:
                        best_success_route_objective_success = 1 - gbm.predict_proba(best_success_route_objective_df[use_features])[:, 1]
                    best_success_route_objective_df["pred_success"] = best_success_route_objective_success
                    average_best_success_route_objective_success = np.average(best_success_route_objective_success)
                    total_best_success_route_objective_distance = np.sum(best_success_route_objective_dict["route_dist"])   
                    
                    # Only consider routes with at least 1 drop 
                    
                    drops_in_route = len(data_route)
                    if drops_in_route > 0:
                            num_drops_day += len(data_route)
                            total_distance_actual_day += total_route_distance_actual
                            total_distance_best_success_objective_day += total_best_success_route_objective_distance
                            success_actual_day_list += data_route["pred_success"].tolist()
                            success_best_success_objective_day_list += best_success_route_objective_df["pred_success"].tolist()
                            total_cost_actual_day += total_route_distance_actual + np.sum(1 - data_route["pred_success"].values) * FAIL_COST
                            total_cost_best_success_objective_day += total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST
                        
                    route_id_count += 1
                    route_id_list.append(route_id_count)
                    route_len_list.append(len(data_route))
                    
                    route_dist_cost_actual_list.append(total_route_distance_actual)
                    route_dist_cost_best_objective_list.append(total_best_success_route_objective_distance)
                    
                    route_failure_cost_actual_list.append(np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                    route_failure_cost_best_objective_list.append(np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)
                    
                    route_total_cost_actual_list.append(total_route_distance_actual+ np.sum(1 - data_route["pred_success"].values) * FAIL_COST)
                    route_total_cost_best_objective_list.append(total_best_success_route_objective_distance + np.sum(1 - best_success_route_objective_df["pred_success"].values) * FAIL_COST)

                num_drops_total += num_drops_day
                total_distance_actual += total_distance_actual_day
                total_distance_best_success_objective += total_distance_best_success_objective_day
                success_actual_list += success_actual_day_list
                success_best_success_objective_list += success_best_success_objective_day_list
                total_cost_actual += total_cost_actual_day
                total_cost_best_success_objective += total_cost_best_success_objective_day  

            # Travel cost 
            round_total_distance_actual = round(total_distance_actual,2)
            
            round_total_distance_best_success_objective = round(total_distance_best_success_objective,2)
            
            # Failure cost
            total_fail_cost_actual = FAIL_COST*num_drops_total*(1-np.average(success_actual_list))
            round_total_fail_cost_actual = round( total_fail_cost_actual,2)
            
            total_fail_cost_best_success_objective = FAIL_COST*num_drops_total*(1-np.average(success_best_success_objective_list))
            round_total_fail_cost_best_success_objective  = round(total_fail_cost_best_success_objective,2)
            
            # Total cost 
            round_total_cost_actual = round(total_cost_actual,2)
            
            round_total_cost_best_success_objective = round(total_cost_best_success_objective,2)

            
            # Difference
            diff_total_distance_cost = round(((round_total_distance_best_success_objective-round_total_distance_actual)/round_total_distance_actual)*100,2)
            
            
            diff_total_fail_cost = round(((round_total_fail_cost_best_success_objective-round_total_fail_cost_actual)/round_total_fail_cost_actual)*100,2)
            
            
            diff_total_cost = round(((round_total_cost_best_success_objective-round_total_cost_actual)/round_total_cost_actual)*100,2)
            
        
            
            #--------------------Bar gragh
            X_set = ['Travel Cost', 'Failure Cost', 'Total Cost']
            Y_actual = [round_total_distance_actual, round_total_fail_cost_actual, round_total_cost_actual]
            Y_best = [round_total_distance_best_success_objective, round_total_fail_cost_best_success_objective, round_total_cost_best_success_objective]
            
            X_axis = np.arange(len(X_set))
            plt.bar(X_axis - 0.2, Y_actual, 0.4, label = 'Baseline')
            for i in range(len(X_axis)):
                plt.text(i,Y_actual[i],Y_actual[i], ha='right',color='green', fontweight='bold')
            
            plt.bar(X_axis + 0.2, Y_best, 0.4, label = 'Optimal')
            for i in range(len(X_axis)):
                plt.text(i,Y_best[i],Y_best[i], ha='left',color='red', fontweight='bold')
            
            plt.xticks(X_axis, X_set)
            plt.xlabel("Cost groups")
            plt.ylabel("Cost ($)")
            plt.title("Costs comparison of CVRP model with/without failure probability")
            plt.legend()
            plt.savefig('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/cost_comparison.png')
            plt.close()
            plot_url2 = '/static/images/cost_comparison.png'
            
            
            # data export 
            route_perf_df = pd.DataFrame()
            route_perf_df["customerid"] = data_day["customerid"]
            route_perf_df["lat"] = data_day["lat"]
            route_perf_df["long"] = data_day["long"]
            route_perf_df["pre_failure_before"] = 1- np.array(success_actual_list)
            route_perf_df["pre_failure_after"] = 1-np.array(success_best_success_objective_list)
            route_perf_df["route"] = data_day["route"]
            route_perf_df.to_csv('D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/result_data.csv', index=False)
            path = '/static/images/result_data.csv'
            
            # Map 
            def get_directions_response(lat1, long1, lat2, long2, mode='drive'):
                url = "https://route-and-directions.p.rapidapi.com/v1/routing"
                key = "165aea8a7fmsh8e6392177d6e3a5p1d9590jsn08d0be6112ff"
                host = "route-and-directions.p.rapidapi.com"
                headers = {"X-RapidAPI-Key": key, "X-RapidAPI-Host": host}
                querystring = {"waypoints":f"{str(lat1)},{str(long1)}|{str(lat2)},{str(long2)}","mode":mode}
                response = requests.request("GET", url, headers=headers, params=querystring)
                return response

            location = pd.read_csv(r"D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/static/images/result_data.csv")

            location = location[["lat", "long"]]
            location_list = location.values.tolist()
            new_list = []
            for i in location_list:
               new_list.append(tuple(i))

            # Depot 
            df_depot = pd.DataFrame()
            depot_lat = [-23.514664500, -23.661295500, -23.433411500, -23.558123100, -23.671398600,-23.526479600,-23.492313700,-23.558123100]
            depot_long = [-46.653909400, -46.487164700, -46.554093100,-46.609302200, -46.716471400,-46.766178900, -46.842962500,-46.609302200]
            depot_name = ["OC01", "OC02", "OC03", "OC04", "OC05", "OC06", "OC09", "OC10"]
            df_depot['depot_name']= depot_name
            df_depot['depot_lat']= depot_lat
            df_depot['depot_long']= depot_long
            depot_location = df_depot[["depot_lat","depot_long"]]
            depot_location_list = depot_location.values.tolist()
            new_depot_list=[]
            for i in depot_location_list:
               new_depot_list.append(tuple(i))

            final_list =  new_list + new_depot_list

            m = folium.Map()
            colors = ['blue','red','green','black','maroon','orange', 'maroon', 'lime', 'teal', 'green']
            for point in new_list[:]:
                folium.Marker(point,icon=folium.Icon( color='blue')).add_to(m)
                
            for point, name in zip (new_depot_list,depot_name) :
                folium.Marker(point,icon=folium.Icon( color='red'),popup=f"Name:<br>{name}").add_to(m)
               
            routes = [[213,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,
                       21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,213],
                      [214,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,
                       54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,214],
                      [215,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,215],
                      [216,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,216],
                      [217,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,217],
                      [217,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,
                       161,162,163,164,165,166,167,168,169,170,171,172,173,174,217],
                      [218,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,
                       197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,218]]
            route_trans = []
            for i in range(len(routes)):
                trans = []
                for shipment_index in routes[i]:
                    trans.append(final_list[shipment_index]) 
                route_trans.append(trans) 
                
                
            responses = []
            for r in range(len( route_trans)):
                for n in range(len(route_trans[r])-1):
                    lat1 = route_trans[r][n][0]
                    lon1 = route_trans[r][n][1]
                    lat2 = route_trans[r][n+1][0]
                    lon2 = route_trans[r][n+1][1]
                    response= get_directions_response(lat1, lon1, lat2, lon2, mode='drive')
                    responses.append(response)
                    mls = response.json()['features'][0]['geometry']['coordinates']
                    points = [(i[1], i[0]) for i in mls[0]]
                    folium.PolyLine(points, weight=5, opacity=1, color=colors[r]).add_to(m)
                    temp = pd.DataFrame(mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
                    df = pd.DataFrame()
                    df = pd.concat([df, temp])
                    sw = df[['Lat', 'Lon']].min().values.tolist()
                    sw = [sw[0]-0.0005, sw[1]-0.0005]
                    ne = df[['Lat', 'Lon']].max().values.tolist()
                    ne = [ne[0]+0.0005, ne[1]+0.0005]
                    m.fit_bounds([sw, ne])  
                 
                             
                # Save map
            mapFname = 'D:/Research Dr. Stanley (MSU)/VRP_Flask/VRP_flask/templates/route_map.html'
            m.save(mapFname)
              
        return render_template('homepage.html', plot_url2=plot_url2,file=path,
                               travelcost_op='${}'.format(round_total_distance_best_success_objective),
                               travelcost_bas='${}'.format(round_total_distance_actual),
                               travelcost_diff= '{}%'.format(diff_total_distance_cost),
                               failurecost_op='${}'.format(round_total_fail_cost_best_success_objective),
                               failurecost_bas='${}'.format(round_total_fail_cost_actual),
                               failurecost_diff='{}%'.format(diff_total_fail_cost),
                               delivercost_op='${}'.format(round_total_cost_best_success_objective),
                               delivercost_bas='${}'.format(round_total_cost_actual),
                               delivercost_diff='{}%'.format(diff_total_cost))

@app.route('/defaul/largescaledata/map')
def route_map():
    return render_template('route_map.html')

        
        






    
    
    
   
    
  

        
        

        
     
       
        
    



            

        



    




    
    
    

    




    
    