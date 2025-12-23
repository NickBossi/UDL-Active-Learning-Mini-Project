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

# Initiating an empy database which will store the accuracy after each acquisition step

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

def create_database(filename):
    if filename == "acq_database":
        bools = ["True", "False"]
        acq_fns = ["entropy","rvar_rat","MI","MSTD", "uniform"]
        runs = [0,1,2]
        
        return {bool_val: {name: {f"run_{number}": [] for number in runs} for name in acq_fns} for bool_val in bools}
    else:
        inference_fns = ["analytic_inf", "MFVI_inf", "uniform"]
        runs = [0,1,2]
        return {name: {f"run_{number}": [] for number in runs} for name in inference_fns}


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
        inf_fn_name, run_num, acq_step, accuracy = data
        run_key = f"run_{run_num}"
        list = database[inf_fn_name][run_key]
    
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

def print_database(filename: str = "acq_database"):
    database = load_database(filename)
    
    if filename == "acq_database":
        for key, value in database.items():
            print(f"Boolean = {key}")
            print("\n*****************************************************\n")
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
        for acq_fn, acq_value in database.items():
            print(f"Acq function = {acq_fn}")
            print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            # database[key][acq_fn] returns a dictionary of run numbers
            for run_num, run_data in database[acq_fn].items():
                print(f"Run number = {run_num}")
                print("_________________________________\n")
                print(run_data)
                print("\n")


def plot_acquisition_curves(file_name='acquisition_curves.png'):
    """
    Draws the acquisition step number vs. accuracy for specified acquisition functions 
    on a single graph using Matplotlib.

    Args:
        acq_names (list): List of acquisition function names (keys in the database).
        database (dict): Dictionary storing the results: 
                         database[name] = [[step_0, acc_0], [step_1, acc_1], ...]
        file_name (str): Name of the file to save the plot.
    """
    plt.figure(figsize=(10, 6))
    database = load_database()          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for acq_fn_name, runs_dict in database["False"].items():
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

            line = plt.plot(steps, mean_acc, label = f"{acq_fn_name} (Mean)")

            #plt.fill_between(steps,
            #                mean_acc - std_acc,
            #
            #                mean_acc + std_acc,
            #                color = line[0].get_color(),
            #                alpha = 0.2)
    ax = plt.gca() 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.title('Accuracy vs. Acquisition Step (Bayesian MC Dropout Mean ± Std)')
    plt.xlabel('Number of Images Acquired')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("plots", file_name), dpi = 900)
    #plt.show()
    plt.close()
    print(f"Plot saved successfully as {file_name}")

def plot_det_vs_non(file_name = "det_vs_non_curve"):
    functions = ["entropy", "var_rat", "BALD"]
    booleans = ["True", "False"]



    database = load_database()          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for fn_name in functions: 
        plt.figure(figsize=(10, 6))
        save_name = f"{file_name}_{fn_name}"
        for boolean in booleans:
            runs_dict = database[boolean][fn_name]

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

                line = plt.plot(steps, mean_acc, label = f"{fn_name} Deterministic = {boolean}")

                plt.fill_between(steps,
                                mean_acc - std_acc,
                                mean_acc + std_acc,
                                color = line[0].get_color(),
                                alpha = 0.2)
        ax = plt.gca() 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
        plt.title(f'Accuracy vs. Acquisition Step for {fn_name} (Bayesian MC Dropout Mean ± Std)')
        plt.xlabel('Number of Images Acquired')
        plt.ylabel('Test Accuracy (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        plt.savefig(os.path.join("plots",save_name), dpi = 900)
        #plt.show()
        plt.close()
        print(f"Plot saved successfully as {save_name}")


def plot_inf_curves(file_name: str = 'inference_curves'):
    plt.figure(figsize=(10, 6))
    database = load_database('inference_database')          #gets datbase from memory
    
    if not database:
        print("Error: The database is empty or None.")
        return

    for acq_fn_name, runs_dict in database.items():
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

            line = plt.plot(steps, mean_acc, label = f"{acq_fn_name} (Mean)")

            plt.fill_between(steps,
                            mean_acc - std_acc,
                            mean_acc + std_acc,
                            color = line[0].get_color(),
                            alpha = 0.2)
    ax = plt.gca() 
    ax.yaxis.set_major_locator(ticker.MultipleLocator(2))
    plt.title('RMSE vs. Acquisition Step ( ± Std)')
    plt.xlabel('Number of Images Acquired')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(os.path.join("plots",file_name), dpi = 900)
    #plt.show()
    plt.close()
    print(f"Plot saved successfully as {file_name}")


#print_database("inference_database")
#plot_inf_curves()
#plot_acquisition_curves()


[[True, 0, 'var_rat', 10, 0.6111], [True, 0, 'var_rat', 20, 0.6804], [True, 0, 'var_rat', 30, 0.7242], [True, 0, 'var_rat', 40, 0.7404], [True, 0, 'var_rat', 50, 0.7638], [True, 0, 'var_rat', 60, 0.7416], [True, 0, 'var_rat', 70, 0.7447], [True, 0, 'var_rat', 80, 0.7494], [True, 0, 'var_rat', 90, 0.7893], [True, 0, 'var_rat', 100, 0.7887], [True, 0, 'var_rat', 110, 0.7541], [True, 0, 'var_rat', 120, 0.7753], [True, 0, 'var_rat', 130, 0.8175], [True, 0, 'var_rat', 140, 0.8], [True, 0, 'var_rat', 150, 0.8327], [True, 0, 'var_rat', 160, 0.8022], [True, 0, 'var_rat', 170, 0.8111], [True, 0, 'var_rat', 180, 0.8206], [True, 0, 'var_rat', 190, 0.8415], [True, 0, 'var_rat', 200, 0.8161], [True, 0, 'var_rat', 210, 0.8288], [True, 0, 'var_rat', 220, 0.8486], [True, 0, 'var_rat', 230, 0.8521], [True, 0, 'var_rat', 240, 0.8406], [True, 0, 'var_rat', 250, 0.8301], [True, 0, 'var_rat', 260, 0.8604], [True, 0, 'var_rat', 270, 0.8499], [True, 0, 'var_rat', 280, 0.8396], [True, 0, 'var_rat', 290, 0.8559], [True, 0, 'var_rat', 300, 0.8664], [True, 0, 'var_rat', 310, 0.8586], [True, 0, 'var_rat', 320, 0.8576], [True, 0, 'var_rat', 330, 0.8557], [True, 0, 'var_rat', 340, 0.8666], [True, 0, 'var_rat', 350, 0.8587], [True, 0, 'var_rat', 360, 0.8619], [True, 0, 'var_rat', 370, 0.8732], [True, 0, 'var_rat', 380, 0.8735], [True, 0, 'var_rat', 390, 0.888], [True, 0, 'var_rat', 400, 0.8817], [True, 0, 'var_rat', 410, 0.8858], [True, 0, 'var_rat', 420, 0.8794], [True, 0, 'var_rat', 430, 0.8794], [True, 0, 'var_rat', 440, 0.8844], [True, 0, 'var_rat', 450, 0.8929], [True, 0, 'var_rat', 460, 0.8919], [True, 0, 'var_rat', 470, 0.898], [True, 0, 'var_rat', 480, 0.8799], [True, 0, 'var_rat', 490, 0.8791], [True, 0, 'var_rat', 500, 0.8837], [True, 0, 'var_rat', 510, 0.9018], [True, 0, 'var_rat', 520, 0.892], [True, 0, 'var_rat', 530, 0.9018], [True, 0, 'var_rat', 540, 0.8976], [True, 0, 'var_rat', 550, 0.9075], [True, 0, 'var_rat', 560, 0.9], [True, 0, 'var_rat', 570, 0.9066], [True, 0, 'var_rat', 580, 0.9008], [True, 0, 'var_rat', 590, 0.8961], [True, 0, 'var_rat', 600, 0.8911], [True, 0, 'var_rat', 610, 0.9052], [True, 0, 'var_rat', 620, 0.9071], [True, 0, 'var_rat', 630, 0.9049], [True, 0, 'var_rat', 640, 0.9206], [True, 0, 'var_rat', 650, 0.9157], [True, 0, 'var_rat', 660, 0.9119], [True, 0, 'var_rat', 670, 0.907], [True, 0, 'var_rat', 680, 0.9184], [True, 0, 'var_rat', 690, 0.9019], [True, 0, 'var_rat', 700, 0.92], [True, 0, 'var_rat', 710, 0.917], [True, 0, 'var_rat', 720, 0.9133], [True, 0, 'var_rat', 730, 0.9169], [True, 0, 'var_rat', 740, 0.9119], [True, 0, 'var_rat', 750, 0.9208], [True, 0, 'var_rat', 760, 0.9244], [True, 0, 'var_rat', 770, 0.9214], [True, 0, 'var_rat', 780, 0.9233], [True, 0, 'var_rat', 790, 0.9244], [True, 0, 'var_rat', 800, 0.9227], [True, 0, 'var_rat', 810, 0.9235], [True, 0, 'var_rat', 820, 0.9306], [True, 0, 'var_rat', 830, 0.9329], [True, 0, 'var_rat', 840, 0.9257], [True, 0, 'var_rat', 850, 0.9272], [True, 0, 'var_rat', 860, 0.9239], [True, 0, 'var_rat', 870, 0.9197], [True, 0, 'var_rat', 880, 0.9305], [True, 0, 'var_rat', 890, 0.9335], [True, 0, 'var_rat', 900, 0.9298], [True, 0, 'var_rat', 910, 0.929], [True, 0, 'var_rat', 920, 0.9296], [True, 0, 'var_rat', 930, 0.9345], [True, 0, 'var_rat', 940, 0.9366], [True, 0, 'var_rat', 950, 0.9344], [True, 0, 'var_rat', 960, 0.9353], [True, 0, 'var_rat', 970, 0.9372], [True, 0, 'var_rat', 980, 0.9274], [True, 0, 'var_rat', 990, 0.9298], [True, 0, 'var_rat', 1000, 0.9366]]

[[True, 1, 'entropy', 10, 0.5218], [True, 1, 'entropy', 20, 0.6031], [True, 1, 'entropy', 30, 0.6243], [True, 1, 'entropy', 40, 0.6272], [True, 1, 'entropy', 50, 0.6533], [True, 1, 'entropy', 60, 0.6619], [True, 1, 'entropy', 70, 0.7159], [True, 1, 'entropy', 80, 0.6907], [True, 1, 'entropy', 90, 0.7407], [True, 1, 'entropy', 100, 0.7362], [True, 1, 'entropy', 110, 0.7564], [True, 1, 'entropy', 120, 0.7633], [True, 1, 'entropy', 130, 0.793], [True, 1, 'entropy', 140, 0.8299], [True, 1, 'entropy', 150, 0.8345], [True, 1, 'entropy', 160, 0.8539], [True, 1, 'entropy', 170, 0.8439], [True, 1, 'entropy', 180, 0.8808], [True, 1, 'entropy', 190, 0.8666], [True, 1, 'entropy', 200, 0.8757], [True, 1, 'entropy', 210, 0.8858], [True, 1, 'entropy', 220, 0.8918], [True, 1, 'entropy', 230, 0.8838], [True, 1, 'entropy', 240, 0.9158], [True, 1, 'entropy', 250, 0.8943], [True, 1, 'entropy', 260, 0.9224], [True, 1, 'entropy', 270, 0.9], [True, 1, 'entropy', 280, 0.9037], [True, 1, 'entropy', 290, 0.9256], [True, 1, 'entropy', 300, 0.9187], [True, 1, 'entropy', 310, 0.9195], [True, 1, 'entropy', 320, 0.9313], [True, 1, 'entropy', 330, 0.934], [True, 1, 'entropy', 340, 0.9383], [True, 1, 'entropy', 350, 0.9375], [True, 1, 'entropy', 360, 0.9273], [True, 1, 'entropy', 370, 0.941], [True, 1, 'entropy', 380, 0.9497], [True, 1, 'entropy', 390, 0.9442], [True, 1, 'entropy', 400, 0.9468], [True, 1, 'entropy', 410, 0.9503], [True, 1, 'entropy', 420, 0.9495], [True, 1, 'entropy', 430, 0.9591], [True, 1, 'entropy', 440, 0.9512], [True, 1, 'entropy', 450, 0.9583], [True, 1, 'entropy', 460, 0.9561], [True, 1, 'entropy', 470, 0.9614], [True, 1, 'entropy', 480, 0.9536], [True, 1, 'entropy', 490, 0.9625], [True, 1, 'entropy', 500, 0.9629], [True, 1, 'entropy', 510, 0.9643], [True, 1, 'entropy', 520, 0.9616], [True, 1, 'entropy', 530, 0.9642], [True, 1, 'entropy', 540, 0.9629], [True, 1, 'entropy', 550, 0.9638], [True, 1, 'entropy', 560, 0.9697], [True, 1, 'entropy', 570, 0.9652], [True, 1, 'entropy', 580, 0.9689], [True, 1, 'entropy', 590, 0.9723], [True, 1, 'entropy', 600, 0.9669], [True, 1, 'entropy', 610, 0.9742], [True, 1, 'entropy', 620, 0.9698], [True, 1, 'entropy', 630, 0.9603], [True, 1, 'entropy', 640, 0.9676], [True, 1, 'entropy', 650, 0.9681], [True, 1, 'entropy', 660, 0.9712], [True, 1, 'entropy', 670, 0.9726], [True, 1, 'entropy', 680, 0.9739], [True, 1, 'entropy', 690, 0.9699], [True, 1, 'entropy', 700, 0.9741], [True, 1, 'entropy', 710, 0.9747], [True, 1, 'entropy', 720, 0.9712], [True, 1, 'entropy', 730, 0.9687], [True, 1, 'entropy', 740, 0.9756], [True, 1, 'entropy', 750, 0.9746], [True, 1, 'entropy', 760, 0.9764], [True, 1, 'entropy', 770, 0.9734], [True, 1, 'entropy', 780, 0.9767], [True, 1, 'entropy', 790, 0.976], [True, 1, 'entropy', 800, 0.9786], [True, 1, 'entropy', 810, 0.9766], [True, 1, 'entropy', 820, 0.9787], [True, 1, 'entropy', 830, 0.9785], [True, 1, 'entropy', 840, 0.976], [True, 1, 'entropy', 850, 0.9791], [True, 1, 'entropy', 860, 0.9769], [True, 1, 'entropy', 870, 0.9811], [True, 1, 'entropy', 880, 0.9716], [True, 1, 'entropy', 890, 0.9766], [True, 1, 'entropy', 900, 0.9781], [True, 1, 'entropy', 910, 0.9787], [True, 1, 'entropy', 920, 0.9796], [True, 1, 'entropy', 930, 0.9794], [True, 1, 'entropy', 940, 0.9796], [True, 1, 'entropy', 950, 0.9821], [True, 1, 'entropy', 960, 0.981], [True, 1, 'entropy', 970, 0.9792], [True, 1, 'entropy', 980, 0.9818], [True, 1, 'entropy', 990, 0.9795], [True, 1, 'entropy', 1000, 0.9825]]

[[True, 2, 'entropy', 10, 0.5349], [True, 2, 'entropy', 20, 0.6012], [True, 2, 'entropy', 30, 0.6154], [True, 2, 'entropy', 40, 0.5993], [True, 2, 'entropy', 50, 0.6577], [True, 2, 'entropy', 60, 0.7158], [True, 2, 'entropy', 70, 0.7095], [True, 2, 'entropy', 80, 0.7388], [True, 2, 'entropy', 90, 0.7428], [True, 2, 'entropy', 100, 0.7543], [True, 2, 'entropy', 110, 0.7963], [True, 2, 'entropy', 120, 0.8314], [True, 2, 'entropy', 130, 0.819], [True, 2, 'entropy', 140, 0.8346], [True, 2, 'entropy', 150, 0.8514], [True, 2, 'entropy', 160, 0.8528], [True, 2, 'entropy', 170, 0.8713], [True, 2, 'entropy', 180, 0.8503], [True, 2, 'entropy', 190, 0.8731], [True, 2, 'entropy', 200, 0.8926], [True, 2, 'entropy', 210, 0.8767], [True, 2, 'entropy', 220, 0.8788], [True, 2, 'entropy', 230, 0.8978], [True, 2, 'entropy', 240, 0.8934], [True, 2, 'entropy', 250, 0.9198], [True, 2, 'entropy', 260, 0.9092], [True, 2, 'entropy', 270, 0.9217], [True, 2, 'entropy', 280, 0.9227], [True, 2, 'entropy', 290, 0.9322], [True, 2, 'entropy', 300, 0.939], [True, 2, 'entropy', 310, 0.931], [True, 2, 'entropy', 320, 0.941], [True, 2, 'entropy', 330, 0.9468], [True, 2, 'entropy', 340, 0.9416], [True, 2, 'entropy', 350, 0.947], [True, 2, 'entropy', 360, 0.9469], [True, 2, 'entropy', 370, 0.9395], [True, 2, 'entropy', 380, 0.9377], [True, 2, 'entropy', 390, 0.9469], [True, 2, 'entropy', 400, 0.9496], [True, 2, 'entropy', 410, 0.9433], [True, 2, 'entropy', 420, 0.9515], [True, 2, 'entropy', 430, 0.9616], [True, 2, 'entropy', 440, 0.9606], [True, 2, 'entropy', 450, 0.9599], [True, 2, 'entropy', 460, 0.9614], [True, 2, 'entropy', 470, 0.9679], [True, 2, 'entropy', 480, 0.9654], [True, 2, 'entropy', 490, 0.966], [True, 2, 'entropy', 500, 0.9659], [True, 2, 'entropy', 510, 0.9646], [True, 2, 'entropy', 520, 0.969], [True, 2, 'entropy', 530, 0.9686], [True, 2, 'entropy', 540, 0.9675], [True, 2, 'entropy', 550, 0.9694], [True, 2, 'entropy', 560, 0.9709], [True, 2, 'entropy', 570, 0.9674], [True, 2, 'entropy', 580, 0.9708], [True, 2, 'entropy', 590, 0.9725], [True, 2, 'entropy', 600, 0.9731], [True, 2, 'entropy', 610, 0.9702], [True, 2, 'entropy', 620, 0.9743], [True, 2, 'entropy', 630, 0.9765], [True, 2, 'entropy', 640, 0.9731], [True, 2, 'entropy', 650, 0.9735], [True, 2, 'entropy', 660, 0.9728], [True, 2, 'entropy', 670, 0.9772], [True, 2, 'entropy', 680, 0.9762], [True, 2, 'entropy', 690, 0.9689], [True, 2, 'entropy', 700, 0.9745], [True, 2, 'entropy', 710, 0.9773], [True, 2, 'entropy', 720, 0.9738], [True, 2, 'entropy', 730, 0.9763], [True, 2, 'entropy', 740, 0.9713], [True, 2, 'entropy', 750, 0.9698], [True, 2, 'entropy', 760, 0.9767], [True, 2, 'entropy', 770, 0.9782], [True, 2, 'entropy', 780, 0.977], [True, 2, 'entropy', 790, 0.9787], [True, 2, 'entropy', 800, 0.9765], [True, 2, 'entropy', 810, 0.9795], [True, 2, 'entropy', 820, 0.9775], [True, 2, 'entropy', 830, 0.9791], [True, 2, 'entropy', 840, 0.9786], [True, 2, 'entropy', 850, 0.9777], [True, 2, 'entropy', 860, 0.9771], [True, 2, 'entropy', 870, 0.9791], [True, 2, 'entropy', 880, 0.9737], [True, 2, 'entropy', 890, 0.9773], [True, 2, 'entropy', 900, 0.9811], [True, 2, 'entropy', 910, 0.9806], [True, 2, 'entropy', 920, 0.9811], [True, 2, 'entropy', 930, 0.9792], [True, 2, 'entropy', 940, 0.9809], [True, 2, 'entropy', 950, 0.9803], [True, 2, 'entropy', 960, 0.9778], [True, 2, 'entropy', 970, 0.9793], [True, 2, 'entropy', 980, 0.9813], [True, 2, 'entropy', 990, 0.9793], [True, 2, 'entropy', 1000, 0.9814]]

[[True, 1, 'BALD', 10, 0.5961], [True, 1, 'BALD', 20, 0.6426], [True, 1, 'BALD', 30, 0.6384], [True, 1, 'BALD', 40, 0.7147], [True, 1, 'BALD', 50, 0.7226], [True, 1, 'BALD', 60, 0.7556], [True, 1, 'BALD', 70, 0.745], [True, 1, 'BALD', 80, 0.7785], [True, 1, 'BALD', 90, 0.7694], [True, 1, 'BALD', 100, 0.7581], [True, 1, 'BALD', 110, 0.7445], [True, 1, 'BALD', 120, 0.7706], [True, 1, 'BALD', 130, 0.7829], [True, 1, 'BALD', 140, 0.7851], [True, 1, 'BALD', 150, 0.7858], [True, 1, 'BALD', 160, 0.7865], [True, 1, 'BALD', 170, 0.79], [True, 1, 'BALD', 180, 0.8128], [True, 1, 'BALD', 190, 0.8052], [True, 1, 'BALD', 200, 0.8378], [True, 1, 'BALD', 210, 0.7978], [True, 1, 'BALD', 220, 0.8113], [True, 1, 'BALD', 230, 0.8058], [True, 1, 'BALD', 240, 0.81], [True, 1, 'BALD', 250, 0.8079], [True, 1, 'BALD', 260, 0.8182], [True, 1, 'BALD', 270, 0.8129], [True, 1, 'BALD', 280, 0.8164], [True, 1, 'BALD', 290, 0.8332], [True, 1, 'BALD', 300, 0.8351], [True, 1, 'BALD', 310, 0.8279], [True, 1, 'BALD', 320, 0.8411], [True, 1, 'BALD', 330, 0.8338], [True, 1, 'BALD', 340, 0.8682], [True, 1, 'BALD', 350, 0.8521], [True, 1, 'BALD', 360, 0.8475], [True, 1, 'BALD', 370, 0.8385], [True, 1, 'BALD', 380, 0.8452], [True, 1, 'BALD', 390, 0.8506], [True, 1, 'BALD', 400, 0.876], [True, 1, 'BALD', 410, 0.8686], [True, 1, 'BALD', 420, 0.8628], [True, 1, 'BALD', 430, 0.8673], [True, 1, 'BALD', 440, 0.876], [True, 1, 'BALD', 450, 0.8682], [True, 1, 'BALD', 460, 0.8619], [True, 1, 'BALD', 470, 0.8679], [True, 1, 'BALD', 480, 0.8682], [True, 1, 'BALD', 490, 0.8717], [True, 1, 'BALD', 500, 0.8797], [True, 1, 'BALD', 510, 0.874], [True, 1, 'BALD', 520, 0.8735], [True, 1, 'BALD', 530, 0.87], [True, 1, 'BALD', 540, 0.8729], [True, 1, 'BALD', 550, 0.8784], [True, 1, 'BALD', 560, 0.8798], [True, 1, 'BALD', 570, 0.8821], [True, 1, 'BALD', 580, 0.8891], [True, 1, 'BALD', 590, 0.8982], [True, 1, 'BALD', 600, 0.8909], [True, 1, 'BALD', 610, 0.8788], [True, 1, 'BALD', 620, 0.8959], [True, 1, 'BALD', 630, 0.8861], [True, 1, 'BALD', 640, 0.897], [True, 1, 'BALD', 650, 0.8988], [True, 1, 'BALD', 660, 0.8992], [True, 1, 'BALD', 670, 0.9014], [True, 1, 'BALD', 680, 0.8954], [True, 1, 'BALD', 690, 0.8968], [True, 1, 'BALD', 700, 0.9047], [True, 1, 'BALD', 710, 0.8912], [True, 1, 'BALD', 720, 0.898], [True, 1, 'BALD', 730, 0.889], [True, 1, 'BALD', 740, 0.8992], [True, 1, 'BALD', 750, 0.9096], [True, 1, 'BALD', 760, 0.91], [True, 1, 'BALD', 770, 0.9092], [True, 1, 'BALD', 780, 0.9195], [True, 1, 'BALD', 790, 0.9199], [True, 1, 'BALD', 800, 0.919], [True, 1, 'BALD', 810, 0.9148], [True, 1, 'BALD', 820, 0.9133], [True, 1, 'BALD', 830, 0.9237], [True, 1, 'BALD', 840, 0.9268], [True, 1, 'BALD', 850, 0.919], [True, 1, 'BALD', 860, 0.9231], [True, 1, 'BALD', 870, 0.9183], [True, 1, 'BALD', 880, 0.9308], [True, 1, 'BALD', 890, 0.9344], [True, 1, 'BALD', 900, 0.9291], [True, 1, 'BALD', 910, 0.9242], [True, 1, 'BALD', 920, 0.9342], [True, 1, 'BALD', 930, 0.9318], [True, 1, 'BALD', 940, 0.9272], [True, 1, 'BALD', 950, 0.9405], [True, 1, 'BALD', 960, 0.9335], [True, 1, 'BALD', 970, 0.9266], [True, 1, 'BALD', 980, 0.9364], [True, 1, 'BALD', 990, 0.9289], [True, 1, 'BALD', 1000, 0.9306]]

[[True, 2, 'BALD', 10, 0.5877], [True, 2, 'BALD', 20, 0.6552], [True, 2, 'BALD', 30, 0.684], [True, 2, 'BALD', 40, 0.6959], [True, 2, 'BALD', 50, 0.7645], [True, 2, 'BALD', 60, 0.7434], [True, 2, 'BALD', 70, 0.7583], [True, 2, 'BALD', 80, 0.7608], [True, 2, 'BALD', 90, 0.7444], [True, 2, 'BALD', 100, 0.7485], [True, 2, 'BALD', 110, 0.6996], [True, 2, 'BALD', 120, 0.7523], [True, 2, 'BALD', 130, 0.761], [True, 2, 'BALD', 140, 0.8061], [True, 2, 'BALD', 150, 0.7733], [True, 2, 'BALD', 160, 0.7868], [True, 2, 'BALD', 170, 0.7895], [True, 2, 'BALD', 180, 0.8158], [True, 2, 'BALD', 190, 0.7842], [True, 2, 'BALD', 200, 0.8082], [True, 2, 'BALD', 210, 0.7896], [True, 2, 'BALD', 220, 0.7849], [True, 2, 'BALD', 230, 0.8126], [True, 2, 'BALD', 240, 0.8206], [True, 2, 'BALD', 250, 0.8044], [True, 2, 'BALD', 260, 0.8273], [True, 2, 'BALD', 270, 0.8163], [True, 2, 'BALD', 280, 0.8245], [True, 2, 'BALD', 290, 0.844], [True, 2, 'BALD', 300, 0.8276], [True, 2, 'BALD', 310, 0.8234], [True, 2, 'BALD', 320, 0.8172], [True, 2, 'BALD', 330, 0.8327], [True, 2, 'BALD', 340, 0.8453], [True, 2, 'BALD', 350, 0.8394], [True, 2, 'BALD', 360, 0.8226], [True, 2, 'BALD', 370, 0.8423], [True, 2, 'BALD', 380, 0.8462], [True, 2, 'BALD', 390, 0.8555], [True, 2, 'BALD', 400, 0.8716], [True, 2, 'BALD', 410, 0.8668], [True, 2, 'BALD', 420, 0.8699], [True, 2, 'BALD', 430, 0.8653], [True, 2, 'BALD', 440, 0.8575], [True, 2, 'BALD', 450, 0.8537], [True, 2, 'BALD', 460, 0.8722], [True, 2, 'BALD', 470, 0.8608], [True, 2, 'BALD', 480, 0.8596], [True, 2, 'BALD', 490, 0.8617], [True, 2, 'BALD', 500, 0.8888], [True, 2, 'BALD', 510, 0.8869], [True, 2, 'BALD', 520, 0.8705], [True, 2, 'BALD', 530, 0.8758], [True, 2, 'BALD', 540, 0.8979], [True, 2, 'BALD', 550, 0.873], [True, 2, 'BALD', 560, 0.8811], [True, 2, 'BALD', 570, 0.8851], [True, 2, 'BALD', 580, 0.8784], [True, 2, 'BALD', 590, 0.8913], [True, 2, 'BALD', 600, 0.8952], [True, 2, 'BALD', 610, 0.8911], [True, 2, 'BALD', 620, 0.8758], [True, 2, 'BALD', 630, 0.893], [True, 2, 'BALD', 640, 0.8897], [True, 2, 'BALD', 650, 0.8919], [True, 2, 'BALD', 660, 0.8929], [True, 2, 'BALD', 670, 0.8987], [True, 2, 'BALD', 680, 0.8991], [True, 2, 'BALD', 690, 0.9012], [True, 2, 'BALD', 700, 0.8931], [True, 2, 'BALD', 710, 0.8929], [True, 2, 'BALD', 720, 0.8935], [True, 2, 'BALD', 730, 0.8884], [True, 2, 'BALD', 740, 0.901], [True, 2, 'BALD', 750, 0.9016], [True, 2, 'BALD', 760, 0.9152], [True, 2, 'BALD', 770, 0.9076], [True, 2, 'BALD', 780, 0.9165], [True, 2, 'BALD', 790, 0.9201], [True, 2, 'BALD', 800, 0.9174], [True, 2, 'BALD', 810, 0.9219], [True, 2, 'BALD', 820, 0.9269], [True, 2, 'BALD', 830, 0.9232], [True, 2, 'BALD', 840, 0.9231], [True, 2, 'BALD', 850, 0.9219], [True, 2, 'BALD', 860, 0.9183], [True, 2, 'BALD', 870, 0.9318], [True, 2, 'BALD', 880, 0.9326], [True, 2, 'BALD', 890, 0.9346], [True, 2, 'BALD', 900, 0.9319], [True, 2, 'BALD', 910, 0.9287], [True, 2, 'BALD', 920, 0.9301], [True, 2, 'BALD', 930, 0.9311], [True, 2, 'BALD', 940, 0.9188], [True, 2, 'BALD', 950, 0.9305], [True, 2, 'BALD', 960, 0.9287], [True, 2, 'BALD', 970, 0.9368], [True, 2, 'BALD', 980, 0.9354], [True, 2, 'BALD', 990, 0.9268], [True, 2, 'BALD', 1000, 0.9328]]


[[False, 1, 'entropy', 10, 0.5464], [False, 1, 'entropy', 20, 0.6113], [False, 1, 'entropy', 30, 0.6334], [False, 1, 'entropy', 40, 0.6335], [False, 1, 'entropy', 50, 0.6646], [False, 1, 'entropy', 60, 0.6914], [False, 1, 'entropy', 70, 0.7313], [False, 1, 'entropy', 80, 0.7564], [False, 1, 'entropy', 90, 0.7709], [False, 1, 'entropy', 100, 0.7972], [False, 1, 'entropy', 110, 0.8079], [False, 1, 'entropy', 120, 0.8406], [False, 1, 'entropy', 130, 0.8027], [False, 1, 'entropy', 140, 0.8297], [False, 1, 'entropy', 150, 0.8446], [False, 1, 'entropy', 160, 0.8368], [False, 1, 'entropy', 170, 0.861], [False, 1, 'entropy', 180, 0.8896], [False, 1, 'entropy', 190, 0.8842], [False, 1, 'entropy', 200, 0.9077], [False, 1, 'entropy', 210, 0.9028], [False, 1, 'entropy', 220, 0.8914], [False, 1, 'entropy', 230, 0.8918], [False, 1, 'entropy', 240, 0.8904], [False, 1, 'entropy', 250, 0.8871], [False, 1, 'entropy', 260, 0.904], [False, 1, 'entropy', 270, 0.9059], [False, 1, 'entropy', 280, 0.9144], [False, 1, 'entropy', 290, 0.9112], [False, 1, 'entropy', 300, 0.9209], [False, 1, 'entropy', 310, 0.9231], [False, 1, 'entropy', 320, 0.9279], [False, 1, 'entropy', 330, 0.9213], [False, 1, 'entropy', 340, 0.9258], [False, 1, 'entropy', 350, 0.9349], [False, 1, 'entropy', 360, 0.9291], [False, 1, 'entropy', 370, 0.9425], [False, 1, 'entropy', 380, 0.9381], [False, 1, 'entropy', 390, 0.9489], [False, 1, 'entropy', 400, 0.9435], [False, 1, 'entropy', 410, 0.9454], [False, 1, 'entropy', 420, 0.9514], [False, 1, 'entropy', 430, 0.9566], [False, 1, 'entropy', 440, 0.9598], [False, 1, 'entropy', 450, 0.9573], [False, 1, 'entropy', 460, 0.9512], [False, 1, 'entropy', 470, 0.9587], [False, 1, 'entropy', 480, 0.957], [False, 1, 'entropy', 490, 0.9637], [False, 1, 'entropy', 500, 0.9574], [False, 1, 'entropy', 510, 0.9593], [False, 1, 'entropy', 520, 0.9575], [False, 1, 'entropy', 530, 0.9547], [False, 1, 'entropy', 540, 0.9665], [False, 1, 'entropy', 550, 0.9531], [False, 1, 'entropy', 560, 0.9589], [False, 1, 'entropy', 570, 0.9705], [False, 1, 'entropy', 580, 0.9699], [False, 1, 'entropy', 590, 0.9712], [False, 1, 'entropy', 600, 0.9694], [False, 1, 'entropy', 610, 0.9676], [False, 1, 'entropy', 620, 0.9682], [False, 1, 'entropy', 630, 0.9658], [False, 1, 'entropy', 640, 0.9676], [False, 1, 'entropy', 650, 0.9731], [False, 1, 'entropy', 660, 0.9682], [False, 1, 'entropy', 670, 0.9742], [False, 1, 'entropy', 680, 0.9712], [False, 1, 'entropy', 690, 0.9725], [False, 1, 'entropy', 700, 0.9691], [False, 1, 'entropy', 710, 0.9737], [False, 1, 'entropy', 720, 0.97], [False, 1, 'entropy', 730, 0.9735], [False, 1, 'entropy', 740, 0.9679], [False, 1, 'entropy', 750, 0.9697], [False, 1, 'entropy', 760, 0.9727], [False, 1, 'entropy', 770, 0.9719], [False, 1, 'entropy', 780, 0.9739], [False, 1, 'entropy', 790, 0.9766], [False, 1, 'entropy', 800, 0.9768], [False, 1, 'entropy', 810, 0.9791], [False, 1, 'entropy', 820, 0.9815], [False, 1, 'entropy', 830, 0.9783], [False, 1, 'entropy', 840, 0.9784], [False, 1, 'entropy', 850, 0.976], [False, 1, 'entropy', 860, 0.9788], [False, 1, 'entropy', 870, 0.9761], [False, 1, 'entropy', 880, 0.976], [False, 1, 'entropy', 890, 0.9798], [False, 1, 'entropy', 900, 0.9782], [False, 1, 'entropy', 910, 0.9805], [False, 1, 'entropy', 920, 0.9805], [False, 1, 'entropy', 930, 0.9791], [False, 1, 'entropy', 940, 0.9808], [False, 1, 'entropy', 950, 0.9814], [False, 1, 'entropy', 960, 0.9821], [False, 1, 'entropy', 970, 0.9764], [False, 1, 'entropy', 980, 0.9791], [False, 1, 'entropy', 990, 0.9801], [False, 1, 'entropy', 1000, 0.979]]


[[False, 2, 'entropy', 10, 0.5941], [False, 2, 'entropy', 20, 0.6688], [False, 2, 'entropy', 30, 0.7092], [False, 2, 'entropy', 40, 0.7439], [False, 2, 'entropy', 50, 0.7302], [False, 2, 'entropy', 60, 0.7389], [False, 2, 'entropy', 70, 0.7851], [False, 2, 'entropy', 80, 0.7798], [False, 2, 'entropy', 90, 0.7782], [False, 2, 'entropy', 100, 0.7893], [False, 2, 'entropy', 110, 0.763], [False, 2, 'entropy', 120, 0.8059], [False, 2, 'entropy', 130, 0.8321], [False, 2, 'entropy', 140, 0.8358], [False, 2, 'entropy', 150, 0.8515], [False, 2, 'entropy', 160, 0.8315], [False, 2, 'entropy', 170, 0.8632], [False, 2, 'entropy', 180, 0.8902], [False, 2, 'entropy', 190, 0.8507], [False, 2, 'entropy', 200, 0.873], [False, 2, 'entropy', 210, 0.9045], [False, 2, 'entropy', 220, 0.8778], [False, 2, 'entropy', 230, 0.8877], [False, 2, 'entropy', 240, 0.8673], [False, 2, 'entropy', 250, 0.9177], [False, 2, 'entropy', 260, 0.9121], [False, 2, 'entropy', 270, 0.8966], [False, 2, 'entropy', 280, 0.9076], [False, 2, 'entropy', 290, 0.9206], [False, 2, 'entropy', 300, 0.9321], [False, 2, 'entropy', 310, 0.933], [False, 2, 'entropy', 320, 0.9213], [False, 2, 'entropy', 330, 0.9295], [False, 2, 'entropy', 340, 0.9358], [False, 2, 'entropy', 350, 0.9403], [False, 2, 'entropy', 360, 0.9517], [False, 2, 'entropy', 370, 0.9485], [False, 2, 'entropy', 380, 0.9524], [False, 2, 'entropy', 390, 0.9425], [False, 2, 'entropy', 400, 0.9598], [False, 2, 'entropy', 410, 0.9567], [False, 2, 'entropy', 420, 0.9584], [False, 2, 'entropy', 430, 0.9562], [False, 2, 'entropy', 440, 0.9584], [False, 2, 'entropy', 450, 0.959], [False, 2, 'entropy', 460, 0.9476], [False, 2, 'entropy', 470, 0.9575], [False, 2, 'entropy', 480, 0.9585], [False, 2, 'entropy', 490, 0.9622], [False, 2, 'entropy', 500, 0.9564], [False, 2, 'entropy', 510, 0.9627], [False, 2, 'entropy', 520, 0.964], [False, 2, 'entropy', 530, 0.9638], [False, 2, 'entropy', 540, 0.9659], [False, 2, 'entropy', 550, 0.9616], [False, 2, 'entropy', 560, 0.9599], [False, 2, 'entropy', 570, 0.9625], [False, 2, 'entropy', 580, 0.9666], [False, 2, 'entropy', 590, 0.9648], [False, 2, 'entropy', 600, 0.965], [False, 2, 'entropy', 610, 0.9698], [False, 2, 'entropy', 620, 0.9707], [False, 2, 'entropy', 630, 0.9632], [False, 2, 'entropy', 640, 0.9653], [False, 2, 'entropy', 650, 0.966], [False, 2, 'entropy', 660, 0.9686], [False, 2, 'entropy', 670, 0.9646], [False, 2, 'entropy', 680, 0.9716], [False, 2, 'entropy', 690, 0.9731], [False, 2, 'entropy', 700, 0.9749], [False, 2, 'entropy', 710, 0.9721], [False, 2, 'entropy', 720, 0.9693], [False, 2, 'entropy', 730, 0.9747], [False, 2, 'entropy', 740, 0.9759], [False, 2, 'entropy', 750, 0.9722], [False, 2, 'entropy', 760, 0.977], [False, 2, 'entropy', 770, 0.9755], [False, 2, 'entropy', 780, 0.9756], [False, 2, 'entropy', 790, 0.9787], [False, 2, 'entropy', 800, 0.9754], [False, 2, 'entropy', 810, 0.9777], [False, 2, 'entropy', 820, 0.9785], [False, 2, 'entropy', 830, 0.9772], [False, 2, 'entropy', 840, 0.979], [False, 2, 'entropy', 850, 0.9805], [False, 2, 'entropy', 860, 0.9801], [False, 2, 'entropy', 870, 0.9783], [False, 2, 'entropy', 880, 0.9804], [False, 2, 'entropy', 890, 0.9774], [False, 2, 'entropy', 900, 0.9797], [False, 2, 'entropy', 910, 0.9745], [False, 2, 'entropy', 920, 0.9786], [False, 2, 'entropy', 930, 0.9817], [False, 2, 'entropy', 940, 0.9827], [False, 2, 'entropy', 950, 0.9778], [False, 2, 'entropy', 960, 0.9804], [False, 2, 'entropy', 970, 0.981], [False, 2, 'entropy', 980, 0.9833], [False, 2, 'entropy', 990, 0.9806], [False, 2, 'entropy', 1000, 0.9824]]

[[False, 1, 'uniform', 10, 0.5887], [False, 1, 'uniform', 20, 0.6126], [False, 1, 'uniform', 30, 0.6351], [False, 1, 'uniform', 40, 0.6664], [False, 1, 'uniform', 50, 0.6937], [False, 1, 'uniform', 60, 0.7353], [False, 1, 'uniform', 70, 0.7685], [False, 1, 'uniform', 80, 0.7781], [False, 1, 'uniform', 90, 0.7592], [False, 1, 'uniform', 100, 0.7783], [False, 1, 'uniform', 110, 0.8241], [False, 1, 'uniform', 120, 0.8223], [False, 1, 'uniform', 130, 0.8247], [False, 1, 'uniform', 140, 0.843], [False, 1, 'uniform', 150, 0.8234], [False, 1, 'uniform', 160, 0.8255], [False, 1, 'uniform', 170, 0.8643], [False, 1, 'uniform', 180, 0.8756], [False, 1, 'uniform', 190, 0.8667], [False, 1, 'uniform', 200, 0.8897], [False, 1, 'uniform', 210, 0.8887], [False, 1, 'uniform', 220, 0.8792], [False, 1, 'uniform', 230, 0.8824], [False, 1, 'uniform', 240, 0.8996], [False, 1, 'uniform', 250, 0.8885], [False, 1, 'uniform', 260, 0.885], [False, 1, 'uniform', 270, 0.8888], [False, 1, 'uniform', 280, 0.898], [False, 1, 'uniform', 290, 0.912], [False, 1, 'uniform', 300, 0.9026], [False, 1, 'uniform', 310, 0.9096], [False, 1, 'uniform', 320, 0.9113], [False, 1, 'uniform', 330, 0.9201], [False, 1, 'uniform', 340, 0.9071], [False, 1, 'uniform', 350, 0.9077], [False, 1, 'uniform', 360, 0.9225], [False, 1, 'uniform', 370, 0.9172], [False, 1, 'uniform', 380, 0.9268], [False, 1, 'uniform', 390, 0.9254], [False, 1, 'uniform', 400, 0.9249], [False, 1, 'uniform', 410, 0.9212], [False, 1, 'uniform', 420, 0.9267], [False, 1, 'uniform', 430, 0.9254], [False, 1, 'uniform', 440, 0.9269], [False, 1, 'uniform', 450, 0.9269], [False, 1, 'uniform', 460, 0.9259], [False, 1, 'uniform', 470, 0.9309], [False, 1, 'uniform', 480, 0.9316], [False, 1, 'uniform', 490, 0.9346], [False, 1, 'uniform', 500, 0.9305], [False, 1, 'uniform', 510, 0.933], [False, 1, 'uniform', 520, 0.9335], [False, 1, 'uniform', 530, 0.9296], [False, 1, 'uniform', 540, 0.9386], [False, 1, 'uniform', 550, 0.937], [False, 1, 'uniform', 560, 0.9321], [False, 1, 'uniform', 570, 0.9348], [False, 1, 'uniform', 580, 0.9419], [False, 1, 'uniform', 590, 0.9385], [False, 1, 'uniform', 600, 0.9387], [False, 1, 'uniform', 610, 0.9389], [False, 1, 'uniform', 620, 0.9395], [False, 1, 'uniform', 630, 0.943], [False, 1, 'uniform', 640, 0.9446], [False, 1, 'uniform', 650, 0.9474], [False, 1, 'uniform', 660, 0.9424], [False, 1, 'uniform', 670, 0.9448], [False, 1, 'uniform', 680, 0.9457], [False, 1, 'uniform', 690, 0.9449], [False, 1, 'uniform', 700, 0.9432], [False, 1, 'uniform', 710, 0.9455], [False, 1, 'uniform', 720, 0.948], [False, 1, 'uniform', 730, 0.946], [False, 1, 'uniform', 740, 0.9491], [False, 1, 'uniform', 750, 0.951], [False, 1, 'uniform', 760, 0.9519], [False, 1, 'uniform', 770, 0.9488], [False, 1, 'uniform', 780, 0.9399], [False, 1, 'uniform', 790, 0.9523], [False, 1, 'uniform', 800, 0.953], [False, 1, 'uniform', 810, 0.9469], [False, 1, 'uniform', 820, 0.95], [False, 1, 'uniform', 830, 0.9543], [False, 1, 'uniform', 840, 0.9517], [False, 1, 'uniform', 850, 0.9547], [False, 1, 'uniform', 860, 0.9555], [False, 1, 'uniform', 870, 0.9558], [False, 1, 'uniform', 880, 0.9534], [False, 1, 'uniform', 890, 0.954], [False, 1, 'uniform', 900, 0.9545], [False, 1, 'uniform', 910, 0.9535], [False, 1, 'uniform', 920, 0.9524], [False, 1, 'uniform', 930, 0.9509], [False, 1, 'uniform', 940, 0.9565], [False, 1, 'uniform', 950, 0.9566], [False, 1, 'uniform', 960, 0.9553], [False, 1, 'uniform', 970, 0.9545], [False, 1, 'uniform', 980, 0.9564], [False, 1, 'uniform', 990, 0.9552], [False, 1, 'uniform', 1000, 0.9579]]

[[False, 2, 'uniform', 10, 0.6385], [False, 2, 'uniform', 20, 0.6724], [False, 2, 'uniform', 30, 0.6817], [False, 2, 'uniform', 40, 0.7194], [False, 2, 'uniform', 50, 0.7482], [False, 2, 'uniform', 60, 0.7762], [False, 2, 'uniform', 70, 0.7936], [False, 2, 'uniform', 80, 0.8248], [False, 2, 'uniform', 90, 0.8097], [False, 2, 'uniform', 100, 0.8337], [False, 2, 'uniform', 110, 0.8481], [False, 2, 'uniform', 120, 0.8331], [False, 2, 'uniform', 130, 0.8629], [False, 2, 'uniform', 140, 0.8489], [False, 2, 'uniform', 150, 0.872], [False, 2, 'uniform', 160, 0.8719], [False, 2, 'uniform', 170, 0.8818], [False, 2, 'uniform', 180, 0.8696], [False, 2, 'uniform', 190, 0.8857], [False, 2, 'uniform', 200, 0.8893], [False, 2, 'uniform', 210, 0.8862], [False, 2, 'uniform', 220, 0.8881], [False, 2, 'uniform', 230, 0.8917], [False, 2, 'uniform', 240, 0.9062], [False, 2, 'uniform', 250, 0.9069], [False, 2, 'uniform', 260, 0.9089], [False, 2, 'uniform', 270, 0.9039], [False, 2, 'uniform', 280, 0.9051], [False, 2, 'uniform', 290, 0.9169], [False, 2, 'uniform', 300, 0.9119], [False, 2, 'uniform', 310, 0.9238], [False, 2, 'uniform', 320, 0.9216], [False, 2, 'uniform', 330, 0.9103], [False, 2, 'uniform', 340, 0.9172], [False, 2, 'uniform', 350, 0.9176], [False, 2, 'uniform', 360, 0.919], [False, 2, 'uniform', 370, 0.9283], [False, 2, 'uniform', 380, 0.9215], [False, 2, 'uniform', 390, 0.9255], [False, 2, 'uniform', 400, 0.9341], [False, 2, 'uniform', 410, 0.9253], [False, 2, 'uniform', 420, 0.9325], [False, 2, 'uniform', 430, 0.9362], [False, 2, 'uniform', 440, 0.9388], [False, 2, 'uniform', 450, 0.9318], [False, 2, 'uniform', 460, 0.9377], [False, 2, 'uniform', 470, 0.9338], [False, 2, 'uniform', 480, 0.9365], [False, 2, 'uniform', 490, 0.9284], [False, 2, 'uniform', 500, 0.9386], [False, 2, 'uniform', 510, 0.9413], [False, 2, 'uniform', 520, 0.9366], [False, 2, 'uniform', 530, 0.9362], [False, 2, 'uniform', 540, 0.9404], [False, 2, 'uniform', 550, 0.9422], [False, 2, 'uniform', 560, 0.943], [False, 2, 'uniform', 570, 0.9418], [False, 2, 'uniform', 580, 0.9475], [False, 2, 'uniform', 590, 0.9434], [False, 2, 'uniform', 600, 0.9378], [False, 2, 'uniform', 610, 0.9358], [False, 2, 'uniform', 620, 0.9467], [False, 2, 'uniform', 630, 0.9474], [False, 2, 'uniform', 640, 0.943], [False, 2, 'uniform', 650, 0.9489], [False, 2, 'uniform', 660, 0.9457], [False, 2, 'uniform', 670, 0.9425], [False, 2, 'uniform', 680, 0.9497], [False, 2, 'uniform', 690, 0.9511], [False, 2, 'uniform', 700, 0.9483], [False, 2, 'uniform', 710, 0.9534], [False, 2, 'uniform', 720, 0.9477], [False, 2, 'uniform', 730, 0.9484], [False, 2, 'uniform', 740, 0.9504], [False, 2, 'uniform', 750, 0.9533], [False, 2, 'uniform', 760, 0.9472], [False, 2, 'uniform', 770, 0.9545], [False, 2, 'uniform', 780, 0.9523], [False, 2, 'uniform', 790, 0.9533], [False, 2, 'uniform', 800, 0.9498], [False, 2, 'uniform', 810, 0.9514], [False, 2, 'uniform', 820, 0.951], [False, 2, 'uniform', 830, 0.9524], [False, 2, 'uniform', 840, 0.9506], [False, 2, 'uniform', 850, 0.953], [False, 2, 'uniform', 860, 0.9529], [False, 2, 'uniform', 870, 0.95], [False, 2, 'uniform', 880, 0.9556], [False, 2, 'uniform', 890, 0.957], [False, 2, 'uniform', 900, 0.9535], [False, 2, 'uniform', 910, 0.9497], [False, 2, 'uniform', 920, 0.9591], [False, 2, 'uniform', 930, 0.9511], [False, 2, 'uniform', 940, 0.954], [False, 2, 'uniform', 950, 0.9539], [False, 2, 'uniform', 960, 0.9565], [False, 2, 'uniform', 970, 0.9565], [False, 2, 'uniform', 980, 0.9595], [False, 2, 'uniform', 990, 0.9549], [False, 2, 'uniform', 1000, 0.9607]]

[[False, 1, 'BALD', 10, 0.5835], [False, 1, 'BALD', 20, 0.6422], [False, 1, 'BALD', 30, 0.685], [False, 1, 'BALD', 40, 0.6727], [False, 1, 'BALD', 50, 0.778], [False, 1, 'BALD', 60, 0.7975], [False, 1, 'BALD', 70, 0.8113], [False, 1, 'BALD', 80, 0.8555], [False, 1, 'BALD', 90, 0.8501], [False, 1, 'BALD', 100, 0.8611], [False, 1, 'BALD', 110, 0.8192], [False, 1, 'BALD', 120, 0.8636], [False, 1, 'BALD', 130, 0.8572], [False, 1, 'BALD', 140, 0.8603], [False, 1, 'BALD', 150, 0.878], [False, 1, 'BALD', 160, 0.8651], [False, 1, 'BALD', 170, 0.8997], [False, 1, 'BALD', 180, 0.8994], [False, 1, 'BALD', 190, 0.9197], [False, 1, 'BALD', 200, 0.9195], [False, 1, 'BALD', 210, 0.9121], [False, 1, 'BALD', 220, 0.9055], [False, 1, 'BALD', 230, 0.8998], [False, 1, 'BALD', 240, 0.9091], [False, 1, 'BALD', 250, 0.9161], [False, 1, 'BALD', 260, 0.9327], [False, 1, 'BALD', 270, 0.9236], [False, 1, 'BALD', 280, 0.9394], [False, 1, 'BALD', 290, 0.9169], [False, 1, 'BALD', 300, 0.9347], [False, 1, 'BALD', 310, 0.9406], [False, 1, 'BALD', 320, 0.936], [False, 1, 'BALD', 330, 0.9321], [False, 1, 'BALD', 340, 0.9074], [False, 1, 'BALD', 350, 0.933], [False, 1, 'BALD', 360, 0.939], [False, 1, 'BALD', 370, 0.9358], [False, 1, 'BALD', 380, 0.9481], [False, 1, 'BALD', 390, 0.9432], [False, 1, 'BALD', 400, 0.9538], [False, 1, 'BALD', 410, 0.9414], [False, 1, 'BALD', 420, 0.9416], [False, 1, 'BALD', 430, 0.9524], [False, 1, 'BALD', 440, 0.9535], [False, 1, 'BALD', 450, 0.9533], [False, 1, 'BALD', 460, 0.9558], [False, 1, 'BALD', 470, 0.9609], [False, 1, 'BALD', 480, 0.9593], [False, 1, 'BALD', 490, 0.9567], [False, 1, 'BALD', 500, 0.9528], [False, 1, 'BALD', 510, 0.9605], [False, 1, 'BALD', 520, 0.9578], [False, 1, 'BALD', 530, 0.9621], [False, 1, 'BALD', 540, 0.9663], [False, 1, 'BALD', 550, 0.9598], [False, 1, 'BALD', 560, 0.9588], [False, 1, 'BALD', 570, 0.9629], [False, 1, 'BALD', 580, 0.9652], [False, 1, 'BALD', 590, 0.9662], [False, 1, 'BALD', 600, 0.9633], [False, 1, 'BALD', 610, 0.9648], [False, 1, 'BALD', 620, 0.9706], [False, 1, 'BALD', 630, 0.97], [False, 1, 'BALD', 640, 0.9684], [False, 1, 'BALD', 650, 0.9654], [False, 1, 'BALD', 660, 0.9621], [False, 1, 'BALD', 670, 0.9696], [False, 1, 'BALD', 680, 0.972], [False, 1, 'BALD', 690, 0.9741], [False, 1, 'BALD', 700, 0.9717], [False, 1, 'BALD', 710, 0.9717], [False, 1, 'BALD', 720, 0.9733], [False, 1, 'BALD', 730, 0.975], [False, 1, 'BALD', 740, 0.9753], [False, 1, 'BALD', 750, 0.9731], [False, 1, 'BALD', 760, 0.9724], [False, 1, 'BALD', 770, 0.9749], [False, 1, 'BALD', 780, 0.979], [False, 1, 'BALD', 790, 0.9742], [False, 1, 'BALD', 800, 0.9767], [False, 1, 'BALD', 810, 0.9768], [False, 1, 'BALD', 820, 0.9797], [False, 1, 'BALD', 830, 0.9756], [False, 1, 'BALD', 840, 0.9753], [False, 1, 'BALD', 850, 0.9763], [False, 1, 'BALD', 860, 0.978], [False, 1, 'BALD', 870, 0.9786], [False, 1, 'BALD', 880, 0.9783], [False, 1, 'BALD', 890, 0.9804], [False, 1, 'BALD', 900, 0.9768], [False, 1, 'BALD', 910, 0.9798], [False, 1, 'BALD', 920, 0.9754], [False, 1, 'BALD', 930, 0.9794], [False, 1, 'BALD', 940, 0.9792], [False, 1, 'BALD', 950, 0.9792], [False, 1, 'BALD', 960, 0.977], [False, 1, 'BALD', 970, 0.9791], [False, 1, 'BALD', 980, 0.9791], [False, 1, 'BALD', 990, 0.9786], [False, 1, 'BALD', 1000, 0.9795]]


[[False, 2, 'BALD', 10, 0.552], [False, 2, 'BALD', 20, 0.5753], [False, 2, 'BALD', 30, 0.6255], [False, 2, 'BALD', 40, 0.7022], [False, 2, 'BALD', 50, 0.7031], [False, 2, 'BALD', 60, 0.7324], [False, 2, 'BALD', 70, 0.7672], [False, 2, 'BALD', 80, 0.7889], [False, 2, 'BALD', 90, 0.7837], [False, 2, 'BALD', 100, 0.8327], [False, 2, 'BALD', 110, 0.757], [False, 2, 'BALD', 120, 0.8615], [False, 2, 'BALD', 130, 0.8574], [False, 2, 'BALD', 140, 0.8225], [False, 2, 'BALD', 150, 0.8569], [False, 2, 'BALD', 160, 0.8947], [False, 2, 'BALD', 170, 0.8707], [False, 2, 'BALD', 180, 0.8916], [False, 2, 'BALD', 190, 0.8998], [False, 2, 'BALD', 200, 0.9256], [False, 2, 'BALD', 210, 0.91], [False, 2, 'BALD', 220, 0.9196], [False, 2, 'BALD', 230, 0.9252], [False, 2, 'BALD', 240, 0.9283], [False, 2, 'BALD', 250, 0.925], [False, 2, 'BALD', 260, 0.9266], [False, 2, 'BALD', 270, 0.9313], [False, 2, 'BALD', 280, 0.9391], [False, 2, 'BALD', 290, 0.9407], [False, 2, 'BALD', 300, 0.9473], [False, 2, 'BALD', 310, 0.9439], [False, 2, 'BALD', 320, 0.9477], [False, 2, 'BALD', 330, 0.9371], [False, 2, 'BALD', 340, 0.9381], [False, 2, 'BALD', 350, 0.9371], [False, 2, 'BALD', 360, 0.9403], [False, 2, 'BALD', 370, 0.9407], [False, 2, 'BALD', 380, 0.9488], [False, 2, 'BALD', 390, 0.9519], [False, 2, 'BALD', 400, 0.9429], [False, 2, 'BALD', 410, 0.9463], [False, 2, 'BALD', 420, 0.9527], [False, 2, 'BALD', 430, 0.9534], [False, 2, 'BALD', 440, 0.9487], [False, 2, 'BALD', 450, 0.9521], [False, 2, 'BALD', 460, 0.9536], [False, 2, 'BALD', 470, 0.9613], [False, 2, 'BALD', 480, 0.9533], [False, 2, 'BALD', 490, 0.9562], [False, 2, 'BALD', 500, 0.9576], [False, 2, 'BALD', 510, 0.9601], [False, 2, 'BALD', 520, 0.9556], [False, 2, 'BALD', 530, 0.9646], [False, 2, 'BALD', 540, 0.9639], [False, 2, 'BALD', 550, 0.9634], [False, 2, 'BALD', 560, 0.9653], [False, 2, 'BALD', 570, 0.9671], [False, 2, 'BALD', 580, 0.9651], [False, 2, 'BALD', 590, 0.9685], [False, 2, 'BALD', 600, 0.9684], [False, 2, 'BALD', 610, 0.9701], [False, 2, 'BALD', 620, 0.9687], [False, 2, 'BALD', 630, 0.9622], [False, 2, 'BALD', 640, 0.9707], [False, 2, 'BALD', 650, 0.9712], [False, 2, 'BALD', 660, 0.9747], [False, 2, 'BALD', 670, 0.9742], [False, 2, 'BALD', 680, 0.974], [False, 2, 'BALD', 690, 0.9729], [False, 2, 'BALD', 700, 0.9734], [False, 2, 'BALD', 710, 0.97], [False, 2, 'BALD', 720, 0.9732], [False, 2, 'BALD', 730, 0.9764], [False, 2, 'BALD', 740, 0.9719], [False, 2, 'BALD', 750, 0.9701], [False, 2, 'BALD', 760, 0.9769], [False, 2, 'BALD', 770, 0.9772], [False, 2, 'BALD', 780, 0.9768], [False, 2, 'BALD', 790, 0.973], [False, 2, 'BALD', 800, 0.9737], [False, 2, 'BALD', 810, 0.9773], [False, 2, 'BALD', 820, 0.9758], [False, 2, 'BALD', 830, 0.9753], [False, 2, 'BALD', 840, 0.9782], [False, 2, 'BALD', 850, 0.9758], [False, 2, 'BALD', 860, 0.9785], [False, 2, 'BALD', 870, 0.9759], [False, 2, 'BALD', 880, 0.9748], [False, 2, 'BALD', 890, 0.9778], [False, 2, 'BALD', 900, 0.9778], [False, 2, 'BALD', 910, 0.9782], [False, 2, 'BALD', 920, 0.9792], [False, 2, 'BALD', 930, 0.9805], [False, 2, 'BALD', 940, 0.979], [False, 2, 'BALD', 950, 0.9789], [False, 2, 'BALD', 960, 0.9797], [False, 2, 'BALD', 970, 0.9761], [False, 2, 'BALD', 980, 0.9792], [False, 2, 'BALD', 990, 0.9791], [False, 2, 'BALD', 1000, 0.98]]


[[False, 1, 'var_rat', 10, 0.5486], [False, 1, 'var_rat', 20, 0.6123], [False, 1, 'var_rat', 30, 0.6519], [False, 1, 'var_rat', 40, 0.7148], [False, 1, 'var_rat', 50, 0.7429], [False, 1, 'var_rat', 60, 0.7823], [False, 1, 'var_rat', 70, 0.808], [False, 1, 'var_rat', 80, 0.8121], [False, 1, 'var_rat', 90, 0.821], [False, 1, 'var_rat', 100, 0.8324], [False, 1, 'var_rat', 110, 0.8374], [False, 1, 'var_rat', 120, 0.8575], [False, 1, 'var_rat', 130, 0.8774], [False, 1, 'var_rat', 140, 0.8887], [False, 1, 'var_rat', 150, 0.8929], [False, 1, 'var_rat', 160, 0.8907], [False, 1, 'var_rat', 170, 0.8947], [False, 1, 'var_rat', 180, 0.9165], [False, 1, 'var_rat', 190, 0.9151], [False, 1, 'var_rat', 200, 0.9101], [False, 1, 'var_rat', 210, 0.9154], [False, 1, 'var_rat', 220, 0.9092], [False, 1, 'var_rat', 230, 0.916], [False, 1, 'var_rat', 240, 0.916], [False, 1, 'var_rat', 250, 0.9397], [False, 1, 'var_rat', 260, 0.9193], [False, 1, 'var_rat', 270, 0.9288], [False, 1, 'var_rat', 280, 0.9263], [False, 1, 'var_rat', 290, 0.9354], [False, 1, 'var_rat', 300, 0.9296], [False, 1, 'var_rat', 310, 0.9332], [False, 1, 'var_rat', 320, 0.9411], [False, 1, 'var_rat', 330, 0.933], [False, 1, 'var_rat', 340, 0.9439], [False, 1, 'var_rat', 350, 0.9448], [False, 1, 'var_rat', 360, 0.9525], [False, 1, 'var_rat', 370, 0.9428], [False, 1, 'var_rat', 380, 0.9589], [False, 1, 'var_rat', 390, 0.9596], [False, 1, 'var_rat', 400, 0.9595], [False, 1, 'var_rat', 410, 0.9599], [False, 1, 'var_rat', 420, 0.9625], [False, 1, 'var_rat', 430, 0.9665], [False, 1, 'var_rat', 440, 0.9653], [False, 1, 'var_rat', 450, 0.9677], [False, 1, 'var_rat', 460, 0.9656], [False, 1, 'var_rat', 470, 0.9666], [False, 1, 'var_rat', 480, 0.9723], [False, 1, 'var_rat', 490, 0.9707], [False, 1, 'var_rat', 500, 0.9672], [False, 1, 'var_rat', 510, 0.9702], [False, 1, 'var_rat', 520, 0.9724], [False, 1, 'var_rat', 530, 0.9731], [False, 1, 'var_rat', 540, 0.9714], [False, 1, 'var_rat', 550, 0.9764], [False, 1, 'var_rat', 560, 0.9686], [False, 1, 'var_rat', 570, 0.9733], [False, 1, 'var_rat', 580, 0.9756], [False, 1, 'var_rat', 590, 0.9722], [False, 1, 'var_rat', 600, 0.9735], [False, 1, 'var_rat', 610, 0.9752], [False, 1, 'var_rat', 620, 0.9772], [False, 1, 'var_rat', 630, 0.9738], [False, 1, 'var_rat', 640, 0.9751], [False, 1, 'var_rat', 650, 0.9762], [False, 1, 'var_rat', 660, 0.9756], [False, 1, 'var_rat', 670, 0.9768], [False, 1, 'var_rat', 680, 0.9767], [False, 1, 'var_rat', 690, 0.9794], [False, 1, 'var_rat', 700, 0.9775], [False, 1, 'var_rat', 710, 0.9786], [False, 1, 'var_rat', 720, 0.9775], [False, 1, 'var_rat', 730, 0.9812], [False, 1, 'var_rat', 740, 0.9767], [False, 1, 'var_rat', 750, 0.9757], [False, 1, 'var_rat', 760, 0.976], [False, 1, 'var_rat', 770, 0.9795], [False, 1, 'var_rat', 780, 0.9796], [False, 1, 'var_rat', 790, 0.9752], [False, 1, 'var_rat', 800, 0.9792], [False, 1, 'var_rat', 810, 0.9808], [False, 1, 'var_rat', 820, 0.9805], [False, 1, 'var_rat', 830, 0.9792], [False, 1, 'var_rat', 840, 0.9787], [False, 1, 'var_rat', 850, 0.9804], [False, 1, 'var_rat', 860, 0.9804], [False, 1, 'var_rat', 870, 0.9805], [False, 1, 'var_rat', 880, 0.977], [False, 1, 'var_rat', 890, 0.9773], [False, 1, 'var_rat', 900, 0.9807], [False, 1, 'var_rat', 910, 0.982], [False, 1, 'var_rat', 920, 0.9783], [False, 1, 'var_rat', 930, 0.981], [False, 1, 'var_rat', 940, 0.9814], [False, 1, 'var_rat', 950, 0.9826], [False, 1, 'var_rat', 960, 0.9826], [False, 1, 'var_rat', 970, 0.9815], [False, 1, 'var_rat', 980, 0.9833], [False, 1, 'var_rat', 990, 0.9802], [False, 1, 'var_rat', 1000, 0.9831]]

run = [[False, 2, 'var_rat', 10, 0.5672], [False, 2, 'var_rat', 20, 0.641], [False, 2, 'var_rat', 30, 0.6834], [False, 2, 'var_rat', 40, 0.6896], [False, 2, 'var_rat', 50, 0.7364], [False, 2, 'var_rat', 60, 0.7801], [False, 2, 'var_rat', 70, 0.7694], [False, 2, 'var_rat', 80, 0.7985], [False, 2, 'var_rat', 90, 0.8096], [False, 2, 'var_rat', 100, 0.8369], [False, 2, 'var_rat', 110, 0.8026], [False, 2, 'var_rat', 120, 0.8983], [False, 2, 'var_rat', 130, 0.8927], [False, 2, 'var_rat', 140, 0.8598], [False, 2, 'var_rat', 150, 0.9024], [False, 2, 'var_rat', 160, 0.873], [False, 2, 'var_rat', 170, 0.9021], [False, 2, 'var_rat', 180, 0.8795], [False, 2, 'var_rat', 190, 0.9074], [False, 2, 'var_rat', 200, 0.9209], [False, 2, 'var_rat', 210, 0.9329], [False, 2, 'var_rat', 220, 0.9316], [False, 2, 'var_rat', 230, 0.9358], [False, 2, 'var_rat', 240, 0.908], [False, 2, 'var_rat', 250, 0.9464], [False, 2, 'var_rat', 260, 0.9481], [False, 2, 'var_rat', 270, 0.9516], [False, 2, 'var_rat', 280, 0.9432], [False, 2, 'var_rat', 290, 0.949], [False, 2, 'var_rat', 300, 0.9489], [False, 2, 'var_rat', 310, 0.9588], [False, 2, 'var_rat', 320, 0.9585], [False, 2, 'var_rat', 330, 0.9602], [False, 2, 'var_rat', 340, 0.9594], [False, 2, 'var_rat', 350, 0.9605], [False, 2, 'var_rat', 360, 0.9588], [False, 2, 'var_rat', 370, 0.9507], [False, 2, 'var_rat', 380, 0.955], [False, 2, 'var_rat', 390, 0.9572], [False, 2, 'var_rat', 400, 0.9643], [False, 2, 'var_rat', 410, 0.9655], [False, 2, 'var_rat', 420, 0.9632], [False, 2, 'var_rat', 430, 0.9617], [False, 2, 'var_rat', 440, 0.9588], [False, 2, 'var_rat', 450, 0.9609], [False, 2, 'var_rat', 460, 0.965], [False, 2, 'var_rat', 470, 0.9634], [False, 2, 'var_rat', 480, 0.9711], [False, 2, 'var_rat', 490, 0.9663], [False, 2, 'var_rat', 500, 0.9682], [False, 2, 'var_rat', 510, 0.9634], [False, 2, 'var_rat', 520, 0.9722], [False, 2, 'var_rat', 530, 0.9665], [False, 2, 'var_rat', 540, 0.9709], [False, 2, 'var_rat', 550, 0.9712], [False, 2, 'var_rat', 560, 0.9753], [False, 2, 'var_rat', 570, 0.9758], [False, 2, 'var_rat', 580, 0.9734], [False, 2, 'var_rat', 590, 0.9697], [False, 2, 'var_rat', 600, 0.9753], [False, 2, 'var_rat', 610, 0.9765], [False, 2, 'var_rat', 620, 0.9733], [False, 2, 'var_rat', 630, 0.9713], [False, 2, 'var_rat', 640, 0.9756], [False, 2, 'var_rat', 650, 0.9751], [False, 2, 'var_rat', 660, 0.9781], [False, 2, 'var_rat', 670, 0.9762], [False, 2, 'var_rat', 680, 0.9785], [False, 2, 'var_rat', 690, 0.9792], [False, 2, 'var_rat', 700, 0.9789], [False, 2, 'var_rat', 710, 0.9793], [False, 2, 'var_rat', 720, 0.9787], [False, 2, 'var_rat', 730, 0.9788], [False, 2, 'var_rat', 740, 0.9797], [False, 2, 'var_rat', 750, 0.9796], [False, 2, 'var_rat', 760, 0.9795], [False, 2, 'var_rat', 770, 0.9799], [False, 2, 'var_rat', 780, 0.9814], [False, 2, 'var_rat', 790, 0.9789], [False, 2, 'var_rat', 800, 0.9768], [False, 2, 'var_rat', 810, 0.9792], [False, 2, 'var_rat', 820, 0.9803], [False, 2, 'var_rat', 830, 0.9793], [False, 2, 'var_rat', 840, 0.9795], [False, 2, 'var_rat', 850, 0.9789], [False, 2, 'var_rat', 860, 0.9793], [False, 2, 'var_rat', 870, 0.9793], [False, 2, 'var_rat', 880, 0.9788], [False, 2, 'var_rat', 890, 0.9811], [False, 2, 'var_rat', 900, 0.9797], [False, 2, 'var_rat', 910, 0.9814], [False, 2, 'var_rat', 920, 0.9805], [False, 2, 'var_rat', 930, 0.9807], [False, 2, 'var_rat', 940, 0.9839], [False, 2, 'var_rat', 950, 0.9799], [False, 2, 'var_rat', 960, 0.9812], [False, 2, 'var_rat', 970, 0.9829], [False, 2, 'var_rat', 980, 0.9815], [False, 2, 'var_rat', 990, 0.9816], [False, 2, 'var_rat', 1000, 0.9827]]


add_run(run)

plot_det_vs_non()
plot_acquisition_curves()
#print_database()

#run = [[True, 1, 'var_rat', 10, 0.5418], [True, 1, 'var_rat', 20, 0.6409], [True, 1, 'var_rat', 30, 0.6479], [True, 1, 'var_rat', 40, 0.7068], [True, 1, 'var_rat', 50, 0.7591], [True, 1, 'var_rat', 60, 0.7635], [True, 1, 'var_rat', 70, 0.7586], [True, 1, 'var_rat', 80, 0.7788], [True, 1, 'var_rat', 90, 0.7655], [True, 1, 'var_rat', 100, 0.7651], [True, 1, 'var_rat', 110, 0.7638], [True, 1, 'var_rat', 120, 0.7655], [True, 1, 'var_rat', 130, 0.783], [True, 1, 'var_rat', 140, 0.7782], [True, 1, 'var_rat', 150, 0.792], [True, 1, 'var_rat', 160, 0.7972], [True, 1, 'var_rat', 170, 0.7942], [True, 1, 'var_rat', 180, 0.8094], [True, 1, 'var_rat', 190, 0.7994], [True, 1, 'var_rat', 200, 0.8039], [True, 1, 'var_rat', 210, 0.8173], [True, 1, 'var_rat', 220, 0.7936], [True, 1, 'var_rat', 230, 0.8221], [True, 1, 'var_rat', 240, 0.7966], [True, 1, 'var_rat', 250, 0.8327], [True, 1, 'var_rat', 260, 0.8187], [True, 1, 'var_rat', 270, 0.7878], [True, 1, 'var_rat', 280, 0.7934], [True, 1, 'var_rat', 290, 0.8415], [True, 1, 'var_rat', 300, 0.8438], [True, 1, 'var_rat', 310, 0.819], [True, 1, 'var_rat', 320, 0.8246], [True, 1, 'var_rat', 330, 0.8356], [True, 1, 'var_rat', 340, 0.8456], [True, 1, 'var_rat', 350, 0.8489], [True, 1, 'var_rat', 360, 0.8154], [True, 1, 'var_rat', 370, 0.8436], [True, 1, 'var_rat', 380, 0.8479], [True, 1, 'var_rat', 390, 0.843], [True, 1, 'var_rat', 400, 0.8596], [True, 1, 'var_rat', 410, 0.8636], [True, 1, 'var_rat', 420, 0.8612], [True, 1, 'var_rat', 430, 0.8492], [True, 1, 'var_rat', 440, 0.8576], [True, 1, 'var_rat', 450, 0.8576], [True, 1, 'var_rat', 460, 0.8487], [True, 1, 'var_rat', 470, 0.8671], [True, 1, 'var_rat', 480, 0.8741], [True, 1, 'var_rat', 490, 0.858], [True, 1, 'var_rat', 500, 0.8746], [True, 1, 'var_rat', 510, 0.8707], [True, 1, 'var_rat', 520, 0.8737], [True, 1, 'var_rat', 530, 0.8843], [True, 1, 'var_rat', 540, 0.8684], [True, 1, 'var_rat', 550, 0.8829], [True, 1, 'var_rat', 560, 0.8858], [True, 1, 'var_rat', 570, 0.8862], [True, 1, 'var_rat', 580, 0.8802], [True, 1, 'var_rat', 590, 0.8763], [True, 1, 'var_rat', 600, 0.8833], [True, 1, 'var_rat', 610, 0.8761], [True, 1, 'var_rat', 620, 0.8935], [True, 1, 'var_rat', 630, 0.8876], [True, 1, 'var_rat', 640, 0.8971], [True, 1, 'var_rat', 650, 0.8942], [True, 1, 'var_rat', 660, 0.9023], [True, 1, 'var_rat', 670, 0.8856], [True, 1, 'var_rat', 680, 0.891], [True, 1, 'var_rat', 690, 0.8914], [True, 1, 'var_rat', 700, 0.901], [True, 1, 'var_rat', 710, 0.8942], [True, 1, 'var_rat', 720, 0.8769], [True, 1, 'var_rat', 730, 0.898], [True, 1, 'var_rat', 740, 0.9003], [True, 1, 'var_rat', 750, 0.9081], [True, 1, 'var_rat', 760, 0.9136], [True, 1, 'var_rat', 770, 0.9084], [True, 1, 'var_rat', 780, 0.9139], [True, 1, 'var_rat', 790, 0.9191], [True, 1, 'var_rat', 800, 0.9177], [True, 1, 'var_rat', 810, 0.9228], [True, 1, 'var_rat', 820, 0.9227], [True, 1, 'var_rat', 830, 0.9222], [True, 1, 'var_rat', 840, 0.9232], [True, 1, 'var_rat', 850, 0.9185], [True, 1, 'var_rat', 860, 0.9266], [True, 1, 'var_rat', 870, 0.9195], [True, 1, 'var_rat', 880, 0.9144], [True, 1, 'var_rat', 890, 0.9251], [True, 1, 'var_rat', 900, 0.9295], [True, 1, 'var_rat', 910, 0.9257], [True, 1, 'var_rat', 920, 0.9336], [True, 1, 'var_rat', 930, 0.9304], [True, 1, 'var_rat', 940, 0.9263], [True, 1, 'var_rat', 950, 0.9328], [True, 1, 'var_rat', 960, 0.9345], [True, 1, 'var_rat', 970, 0.9266], [True, 1, 'var_rat', 980, 0.9358], [True, 1, 'var_rat', 990, 0.9317], [True, 1, 'var_rat', 1000, 0.9254]]

#add_run(run)