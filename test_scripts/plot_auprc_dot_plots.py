import numpy as np
from collections import defaultdict
from sklearn.metrics import average_precision_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys

ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"

SPECIES = ["mm10", "hg38", "canFam4", "rn5", "rheMac10", "monDom5", "galGal6"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

model_names_dict = {"mm10" : "Mouse", "hg38" : "Human", "monDom5" : "Possum", "canFam4" : "Dog", "galGal6" : "Chicken", "rn5" : "Rat", "rheMac10" : "Macaque"}

# colors_dict = {"mm10" : "tab:blue", "hg38" : "tab:orange", "monDom5" : "tab:green", "canFam4" : "tab:red", "galGal6" : "tab:purple", "rn5" : "tab:brown", "rheMac10" : "tab:pink"}

model_types = ["DA", "BA", "BM", "Multi", "EM", "Reg"]

model_type_names = ["DA_model", "BA_model", "basic_model", "multi_model", "ensemble_model", "ensemble_model_reg"]

test_species = sys.argv[1]
if test_species in SPECIES_SMALL:
    TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]
else:
    TFS = ["CEBPA"]

def get_preds_file(tf, train_species, test_species, model_type):
    preds_root = ROOT + "/model_out/"
    return preds_root + tf + "_" + model_type + "_" + train_species + "-trained_" + test_species + "-test.preds.npy"

def load_all_test_set_preds(test_species):
    # takes a while to run.
    preds_dict = defaultdict(lambda : dict())

    # loop over mouse-trained, human-trained models, and DA mouse-trained models
    for tf in TFS:
        if tf == "CEBPA":
            all_species = SPECIES
        else:
            all_species = SPECIES_SMALL
        models = all_species + ['multi_model', 'BA_model', 'DA_model', 'ensemble_model', "ensemble_model_reg"]
        for model in models:
            if model in ['DA_model', 'multi_model', 'BA_model', 'ensemble_model', "ensemble_model_reg"]:
                trained = "all-but-" + test_species
                model_type = model
            else:
                trained = model
                model_type = 'basic_model'
            print("=== " + tf + ", " + trained + "-trained ===")

            # load predictions for model runs
            preds_file = get_preds_file(tf, trained, test_species, model_type)
            try:
                preds_dict[model][tf] = np.load(preds_file)
            except:
                print("Could not load preds file:", preds_file)
            
    return preds_dict

def get_test_bed_file(tf, species):
    # This function returns the path to a BED-format file
    # containing the chromosome names, starts, ends, and
    # binding labels for all examples to test the model with.
    # This file is specific to each tf -- the last column
    # should contain the binding label for each window
    return(ROOT + "exp_data/" + species + "/" + tf + "/chr2.bed")


def get_test_labels(tf, species):
    # This function reads in the test-data bed file 
    # for a given species and TF and returns the binding labels
    # for each example in that file.
    
    labels_file = get_test_bed_file(tf, species)
    with open(labels_file) as f:
        return np.array([int(line.split()[-1]) for line in f])

def get_auPRCs(preds, labels):
    # This function calculates the auPRC for each set of
    # predictions passed in. The length of the 2nd axis
    # of the predictions array passed in will be the # of
    # auPRCs returned as a list. The length of the 1st axis
    # of the predictions array should match the length
    # of the labels array.
    assert preds.shape[0] <= len(labels), (preds.shape, len(labels))
    if preds.shape[0] < len(labels):
        labels = labels[:preds.shape[0]]
        
    return [average_precision_score(labels, model_preds) for model_preds in preds.T]


def get_auPRC_df(preds_dict, test_species): 
    # This function loads in binding labels for each TF for 
    # a given test species, and for each TF, calculates the auPRC
    # using each set of predictions that is input in "preds_dict".
    auPRC_dicts = defaultdict(lambda : dict())

    for tf in TFS:
        test_labels = get_test_labels(tf, test_species)
        if tf == "CEBPA":
            all_species = SPECIES
        else:
            all_species = SPECIES_SMALL
        models = preds_dict.keys()
        for model in models:
            if model in all_species + ['multi_model', 'BA_model', 'DA_model', 'ensemble_model', 'ensemble_model_reg']:
                auPRC_dicts[model][tf] = get_auPRCs(preds_dict[model][tf],
                                                    test_labels)
    # before returning all the auPRCs in dictionaries,
    # we just need to reformat how they are stored
    # because seaborn expects particularly formatted input

    return format_data_for_seaborn(auPRC_dicts, test_species)

    
### Plot data preprocessing

def get_model_name(model, test_species):
    if model in SPECIES:
        model_name = model_names_dict[model] + "-trained"
    elif model == 'DA_model':
        model_name = "DA: " + model_names_dict[test_species] + " target"
    elif model == 'BA_model':
        model_name = "BA: " + model_names_dict[test_species] + " target"
    elif model == 'ensemble_model':
        model_name = "Ensemble: " + model_names_dict[test_species] + " target"
    elif model == 'ensemble_model_reg':
        model_name = "Regression: " + model_names_dict[test_species] + " target"
    else:
        model_name = "Multi-species: " + model_names_dict[test_species] + " target"
    return model_name

# def get_model_from_name(model_name):
#     if model_name.endswith("-trained"):
#         species_name = model_name.split("-")[0]
#         model = list(model_names_dict.keys())[list(model_names_dict.values()).index(species_name)]
#     else:
#         model = model_name.split()[1]
#     return model


def format_data_for_seaborn(auPRC_dicts, test_species):
    # This function re-formats the "auPRC_dicts" list of dicts
    # into one pandas DataFrame that matches how seaborn expects
    # data to be input for the plot we will be making
    tf_col = []
    model_col = []
    auprc_col = []
    # assumes reps are constant across training species and TFs
    model_list = list(auPRC_dicts.keys())
    reps = len(auPRC_dicts[model_list[0]][TFS[0]])
    # length = len(auPRC_dicts[test_species][TFS[0]][0])
    
    for tf in TFS:
        if tf == "CEBPA":
            model_list_tmp = model_list
        else:
            model_list_tmp = []
            for mdl in model_list:
                if (mdl not in SPECIES) or (mdl in SPECIES_SMALL):
                    model_list_tmp.append(mdl)
        tf_col.extend([tf] * reps * len(model_list_tmp))
        for model in model_list_tmp:
            model_name = get_model_name(model, test_species)
            model_col.extend([model_name] * reps)
            auprc_col.extend(auPRC_dicts[model][tf])
            # try:
            #     auprc_col.extend(auPRC_dicts[model][tf])
            # except:
            #     auprc_col.extend(np.empty(shape=(reps, 1)))
        
    return pd.DataFrame({"TF":tf_col, "Model":model_col, "auPRC":auprc_col})

# Plotting code

# Constants to specify plot appearance details
DOT_SIZE = 8
FIG_SIZE_UNIT = 10
FIG_SIZE = (FIG_SIZE_UNIT + 1.5, FIG_SIZE_UNIT - 1)
FIG_SIZE_SMALL = (FIG_SIZE_UNIT, FIG_SIZE_UNIT - 1)
colors_dict = {"mm10" : "tab:blue", "hg38" : "tab:orange", "monDom5" : "tab:green", 
               "canFam4" : "tab:red", "galGal6" : "tab:purple", "rn5" : "tab:brown", 
               "rheMac10" : "tab:pink", "DA_model" : "tab:gray", "multi_model": "tab:olive",
               "BA_model":"tab:cyan", "ensemble_model":"k", "ensemble_model_reg":"w"}

big_include = list(colors_dict.keys())
# COLORS = ["#0062B8", "#A2FFB6", "#FF0145", "#FFA600"]
AX_FONTSIZE = 16
AXTICK_FONTSIZE = 13
TITLESIZE = 17

from matplotlib.lines import Line2D


def make_boxplot(df, test_species, save_files = True, include = [],
                 fig_size = FIG_SIZE, colors_to_use = list(colors_dict.values()),
                 dot_size = DOT_SIZE, titlesize = TITLESIZE,
                 ax_fontsize = AX_FONTSIZE,
                 axtick_fontsize = AXTICK_FONTSIZE):
    
    # This function creates one boxplot using seaborn.
    # The data plotted must be stored in a pandas DataFrame (input = "df"),
    # including 3 columns: TF, Species, and auPRC (case-sensitive names).

    # Use the argument save_files to toggle between saving plots
    # and outputting them within the notebook.
    
    # If you want to create a plot containing only a subset of the data
    # in your input DataFrame, specify which training species / model types
    # to include by listing the model types by name in a list and give
    # to the argument "include" (see cell below for examples). Plotting
    # will follow the order of the model types as they are listed in "include".
    
    
    # determine y-axis upper limit of plots
    # this is done before data is subsetted to keep axis consistent
    # regardless of which subset of data is used
    yax_max = max(df["auPRC"]) + 0.15

    label_color_dict = {}
    for item in colors_dict:
        label_color_dict[get_model_name(item, test_species)] = colors_dict[item]
    
    # include should have species to plot in order of how you want them sorted on plot
    if len(include) > 0:
        # models_include = set([model for model in include])
        model_names_include = [get_model_name(model, test_species) for model in include]
        df_to_use = df[[model in model_names_include for model in df["Model"]]]
        cols_list = []
        labels_list = []
        
        for index, color in enumerate(colors_to_use):
            if big_include[index] in include:
                cols_list.append(color)
                labels_list.append(get_model_name(big_include[index], test_species))
        cols = label_color_dict
    else:
        df_to_use = df
        cols_list = colors_to_use
        labels_list = [get_model_name(model, test_species) for model in df["Model"]]
        cols = label_color_dict
    
    sns.set(style = "white")

    # plot individual dots

    ax = sns.stripplot(x = "TF", y = "auPRC", hue = "Model",
                       data = df_to_use,
                       dodge = True,
                       palette = cols,
                       size = dot_size,
                       #edgecolor = "0.0001",
                       linewidth = 0.5)
    
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=species,
                              markeredgecolor='k', markeredgewidth=1,
                          markerfacecolor=c, markersize=10) for c, species in zip(cols_list, labels_list)]

    ax.legend(handles=legend_elements, loc = 'upper right', ncol = 2)

    # add legend
    #ax.legend(loc = 'upper right', ncol = 1, frameon = False)

    # format and label axes
    ax.set_xlabel("", fontsize = 0)
    ax.set_ylabel("Area Under PRC", fontsize = ax_fontsize)
    ax.set_xticklabels(labels = TFS, fontsize = ax_fontsize)
    ax.tick_params(axis='y', which='major', pad = -2, labelsize = axtick_fontsize)
    plt.ylim(0, yax_max) # limit is hard-coded so that it's constant across all plots
    if yax_max < 0.5:
        step = 0.1
    else:
        step = 0.2
    plt.yticks(list(np.arange(0, yax_max, step)))
    
    # modify font size if there isn't much to plot
    if len(include) < 3:
        titlesize = titlesize - 2
    
    # use plot-acceptable version of test data species name
    # e.g. "mm10" --> "Mouse"
    title = "Model Performance, "
    title += r"$\bf{" + model_names_dict[test_species] + "}$"
    title += " Test Data"
    plt.title(title, fontsize = titlesize)
        
    if include is None:
        save_suffix = "all"
    elif ("ensemble_model" not in include):
        if ("DA_model" not in include):
            if ("multi_model" not in include):
                save_suffix = "basic"
            else:
                save_suffix = "multi"
        else:
            save_suffix = "DA"
    else:
        save_suffix = "EM"

    if save_files:
        plt.savefig(ROOT + "plots/dotplots_" + test_species + "_test_" + save_suffix + ".png",
                    bbox_inches='tight', pad_inches = 0.1, dpi = 300)
        # plt.savefig(ROOT + "plots/dotplots_" + test_species + "_test_" + save_suffix + ".pdf",
        #             bbox_inches='tight', pad_inches = 0.1)
        print("DONE!")
    
    plt.clf()


    

all_preds = load_all_test_set_preds(test_species)

test_df = get_auPRC_df(all_preds, test_species)

print(test_df)

test_df.to_csv(ROOT + "plots/" + test_species + "_test_all_auPRCs.csv")

include_list_all = [SPECIES, SPECIES + ['multi_model'], SPECIES + ['multi_model', 'BA_model', 'DA_model'], SPECIES + ['multi_model', 'BA_model', 'DA_model', 'ensemble_model', 'ensemble_model_reg']]

for include_list in include_list_all:
    sns.set(rc = {'figure.figsize' : FIG_SIZE})
    make_boxplot(test_df, test_species, include = include_list)
