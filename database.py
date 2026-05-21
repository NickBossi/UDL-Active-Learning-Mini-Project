import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import itertools
import copy
import random
from IPython.display import Image
import matplotlib.ticker as ticker


# Handling of data during training using each acquisition function 

#Datbase will have form:
# Deteministic: True of False
# acq_fn_name
# run number (0-2)
# [acq_number][accuracy]


import pickle
import os


def get_base_path():
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            print("Mounting Google Drive")
            drive.mount('/content/drive')
        
        print("Running in Colab")
        BASE_PATH = '/content/drive/MyDrive/Oxford/UDL/UDL_mini_project/UDL_results/'

    except ImportError:
        BASE_PATH = './UDL_results/'
        os.makedirs(BASE_PATH, exist_ok=True)

    return BASE_PATH

# Initiating an empy database which will store the accuracy after each acquisition step
def create_database(filename):
    bools = ["True", "False"]
    if filename == "acq_database":
        bools = ["True", "False"]
        acq_fns = ["entropy","var_rat","MI","MSTD", "uniform","var_rat_mod","mean_change"]
        runs = [0,1,2]
        
        return {bool_val: {name: {f"run_{number}": [] for number in runs} for name in acq_fns} for bool_val in bools}
    else:
        inference_fns = ["analytic_inf", "MFVI_inf", "uniform"]
        runs = [0,1,2]
        return {bool_val: {name: {f"run_{number}": [] for number in runs} for name in inference_fns} for bool_val in bools}

# Saves database to path
def save_database(data, filename: str = 'acq_database'):
    base_path = get_base_path()
    full_path = os.path.join(base_path, filename)

    if os.path.exists(full_path):
        existing_data = load_database(filename)

    if not data:
        print("Data is empty. Initialising new database.")
        data = create_database(filename)

    try:
        with open(full_path, 'wb') as f:
            pickle.dump(data, f)
            print(f"Database saved successfully to {full_path}")

    except Exception as e:
        print(f"Error saving database to {full_path}: {e}")

# Loads relevant databse
def load_database(filename: str = 'acq_database'):
    base_path = get_base_path()
    save_path = os.path.join(base_path, filename)
    if os.path.exists(save_path):
        try:
            with open(save_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading database from {filename}: {e}")
            return create_database(filename)
    else:
        print(f"No previous database found at {filename}. Starting fresh.")
        return create_database(filename)
    
# Updates databse with new data tuple
def update_database(data, filename: str = "acq_database"):
    base_path = get_base_path()
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        database = load_database(filename)

    else:
        database = create_database(filename)

    if filename == "acq_database":
        is_det, run_num, acq_fn_name, acq_step, accuracy = data
        det_key = str(is_det)
        run_key = f"run_{run_num}"
        
        list = database[det_key][acq_fn_name][run_key]

    else:
        is_det, inf_fn_name, run_num, acq_step, accuracy = data
        run_key = f"run_{run_num}"
        list = database[str(is_det)][inf_fn_name][run_key]
    
    # First checks if there is data in specific run, if so, overwrites
    found = False
    for i, (prev_acq_step,_) in enumerate(list):
        if prev_acq_step == acq_step:
            list[i] = [acq_step, accuracy]
            found = True
            break
        
    if not found:
        list.append([acq_step, accuracy])

    save_database(database, filename)

# Adds a full run (to help import runs from google colab)
def add_run(run, filename: str = "acq_database"):
    base_path = get_base_path()
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        database = load_database(filename)

    else:
        database = create_database(filename)

    for data in run:
        update_database(data, filename)

def rename_key(oldkey_newkey_tuple_list):
    database = load_database()

    for oknk in oldkey_newkey_tuple_list:
        (oldkey, newkey) = oknk
        for boolean, value in database.items():
            database[boolean][newkey] = database[boolean].pop(oldkey)
        save_database(database)

# Helps view databse
def print_database(filename: str = "acq_database"):
    database = load_database(filename)
    
    if filename == "acq_database":
        for key, value in database.items():
            print("\n**************************************************************************************************************\n")
            print(f"Boolean = {key}")
            print("\n**************************************************************************************************************\n")
            # database[key] returns a dictionary of acquisition functions
            for acq_fn, acq_value in database[key].items():
                print(f"Acq function = {acq_fn}")
                print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                # database[key][acq_fn] returns a dictionary of run numbers
                for run_num, run_data in database[key][acq_fn].items():
                    print(f"Run number = {run_num}")
                    print("_________________________________\n")
                    print(run_data)
                    print("\n")
    else:
        for key, value in database.items():
            print(f"Boolean = {key}")
            print("\n*****************************************************\n")
            for acq_fn, acq_value in database[key].items():
                print(f"Acq function = {acq_fn}")
                print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                # database[key][acq_fn] returns a dictionary of run numbers
                for run_num, run_data in database[key][acq_fn].items():
                    print(f"Run number = {run_num}")
                    print("_________________________________\n")
                    print(run_data)
                    print("\n")

# Renaming for plotting
rename_dict = {"BALD":"BALD", "var_rat": "Var Ratios", "entropy": "Entropy", "Mean_STD": "Mean STD", "uniform": "Random", "analytic_inf": "Analytic Inference", "MFVI_inf": "MFVI", "var_rat_mod":"Modified Var Ratios", "mean_change": "Mean Expected Change"}

# Plots mean of acquisition cutves
def plot_acquisition_curves(file_name='acquisition_curves.png'):

    plt.figure(figsize=(8,6))
    database = load_database()          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for boolean in [True, False]:

        for acq_fn_name, runs_dict in database[str(boolean)].items():
            if boolean and acq_fn_name not in ["mean_change:"]:
                pass
            else:
                run_accuracies = []
                steps = None

                for run_id, data in runs_dict.items():

                    if not data or len(data) ==0:
                        continue
                    
                    data = np.array(data)
                    steps = data[:,0]
                    run_accuracies.append(data[:,1])

                if len(run_accuracies) > 0 and steps is not None:

                    min_len = min(len(r) for r in run_accuracies)

                    run_accuracies = np.array([r[:min_len] for r in run_accuracies])*100

                    steps = steps[:min_len]
                        
                    mean_acc = np.mean(run_accuracies, axis = 0)

                    line = plt.plot(steps, mean_acc, label = f"{rename_dict[acq_fn_name]}")

    ax = plt.gca() 
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_ylim(bottom = 70)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    #plt.xlabel('Number of Images Acquired')
    #plt.ylabel('Test Accuracy (%)')
    plt.legend(loc = "lower right")
    plt.xlim(0, 1000)
    plt.ylim(70,100)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", file_name), dpi = 900)
    #plt.show()
    plt.close()
    print(f"Plot saved successfully as {file_name}")

# Plots deterministic vs non determoinistic runs
def plot_det_vs_non(file_name = "det_vs_non_curve"):
    functions = ["entropy", "var_rat", "BALD", "Mean_STD", "var_rat_mod", "mean_change"]
    booleans = [False, True]

    database = load_database()          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for fn_name in functions: 
        plt.figure(figsize=(10, 6))
        save_name = f"{rename_dict[fn_name]}"
        for boolean in booleans:
            
            label_name = fn_name[:]

            print("fn_name:", fn_name, "| boolean:", boolean)

            # When plotting modified variational ratios, we compare to unmodified variation ratios
            if fn_name == "var_rat_mod":
                if boolean:
                    lookup_name = "var_rat_mod" 
                else:
                    lookup_name = "var_rat"
            else:
                lookup_name = fn_name

            runs_dict = database[str(boolean)][lookup_name]

            run_accuracies = []
            steps = None

            for run_id, data in runs_dict.items():

                if not data or len(data) ==0:
                    continue
                
                data = np.array(data)
                steps = data[:,0]
                run_accuracies.append(data[:,1])

            if len(run_accuracies) > 0 and steps is not None:

                min_len = min(len(r) for r in run_accuracies)

                run_accuracies = np.array([r[:min_len] for r in run_accuracies])*100

                steps = steps[:min_len]
                    
                mean_acc = np.mean(run_accuracies, axis = 0)
                std_acc = np.std(run_accuracies, axis = 0)

                if boolean:
                    col = "blue"
                    line = plt.plot(steps, mean_acc, label = f"Deterministic {rename_dict[label_name]}", color = col)
                else:
                    col = "red"
                    line = plt.plot(steps, mean_acc, label = f"{rename_dict[label_name]}", color = col)
                    

                plt.fill_between(steps,
                                mean_acc - std_acc,
                                mean_acc + std_acc,
                                color = col,
                                alpha = 0.2)
        ax = plt.gca() 

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(100))

        ax.set_ylim(bottom = 70)

        plt.legend(loc = "lower right")
        plt.xlim(0, 1000)
        plt.ylim(70,100)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join("plots",save_name), dpi = 900)
        #plt.show()
        plt.close()
        print(f"Plot saved successfully as {save_name}")

# Plots inference curves
def plot_inf_curves(file_name: str = 'inference_curves'):
    plt.figure(figsize=(10, 6))
    database = load_database('inference_database')          #gets database from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for acq_fn_name, runs_dict in database["True"].items():
        run_accuracies = []
        steps = None

        for run_id, data in runs_dict.items():

            if not data or len(data) ==0:
                continue
            
            data = np.array(data)
            steps = data[:,0]
            run_accuracies.append(data[:,1])

        if len(run_accuracies) > 0 and steps is not None:

            min_len = min(len(r) for r in run_accuracies)

            run_accuracies = np.array([r[:min_len] for r in run_accuracies])

            steps = steps[:min_len]
                
            mean_acc = np.mean(run_accuracies, axis = 0)
            std_acc = np.std(run_accuracies, axis = 0)

            if rename_dict[acq_fn_name] == "MFVI":
                col = "blue"
            else:
                col = "red"

            line = plt.plot(steps, mean_acc, label = f"{rename_dict[acq_fn_name]}", color = col)

    ax = plt.gca() 

    plt.xlabel('Number of Images Acquired')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join("plots",file_name), dpi = 900)
    #plt.show()
    plt.close()
    print(f"Plot saved successfully as {file_name}")

# Plots frozen vs not frosen MFVI curves
def plot_retrain_vs_not(file_name = "retrain_vs_not"):
    functions = ["analytic_inf", "MFVI_inf"]
    booleans = [False, True]



    database = load_database("inference_database")          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return
    
    for fn_name in functions: 
        plt.figure(figsize=(10, 6))
        save_name = f"{file_name}_{rename_dict[fn_name]}"

        for boolean in booleans:
            runs_dict = database[str(boolean)][fn_name]

            run_accuracies = []
            steps = None

            for run_id, data in runs_dict.items():

                if not data or len(data) ==0:
                    continue
                
                data = np.array(data)
                steps = data[:,0]
                run_accuracies.append(data[:,1])

            if len(run_accuracies) > 0 and steps is not None:

                min_len = min(len(r) for r in run_accuracies)

                run_accuracies = np.array([r[:min_len] for r in run_accuracies])

                steps = steps[:min_len]
                    
                mean_acc = np.mean(run_accuracies, axis = 0)
                std_acc = np.std(run_accuracies, axis = 0)

                if boolean:
                    col = "red"
                    line = plt.plot(steps, mean_acc, label = f"Retrained {rename_dict[fn_name]}", color = col)

                else:
                    col = "blue"
                    line = plt.plot(steps, mean_acc, label = f"{rename_dict[fn_name]}", color = col)


                plt.fill_between(steps,
                                mean_acc - std_acc,
                                mean_acc + std_acc,
                                color = col,
                                alpha = 0.2)
        ax = plt.gca() 

        if rename_dict[fn_name] == "MFVI":
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        else:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
        #plt.title(f'RMSE vs. Acquisition Step for {rename_dict[fn_name]}')
        plt.xlabel('Number of Images Acquired')
        plt.ylabel('Test RMSE')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join("plots",save_name), dpi = 900)
        #plt.show()
        plt.close()
        print(f"Plot saved successfully as {save_name}")


plot_retrain_vs_not()