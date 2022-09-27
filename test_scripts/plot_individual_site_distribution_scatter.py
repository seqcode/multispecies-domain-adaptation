from collections import defaultdict
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import sys

ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"

SPECIES = ["mm10", "hg38", "canFam4", "rn5", "rheMac10", "monDom5", "galGal6"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

model_names_dict = {"mm10" : "Mouse", "hg38" : "Human", "monDom5" : "Possum", "canFam4" : "Dog", "galGal6" : "Chicken", "rn5" : "Rat", "rheMac10" : "Macaque"}

model_types = ["DA", "BM"]

model_type_names = ["kelly", "basic_model"]

test_species = sys.argv[2]
train_species = sys.argv[1]
if test_species in SPECIES_SMALL:
    TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]
else:
    TFS = ["CEBPA"]

big_include = SPECIES + [spec + "_DA" for spec in SPECIES]

DOT_SIZE = 5
ALPHA = 0.03
AXIS_SIZE = 11
AX_OFFSET = 0.02
TF_TWINAX_OFFSET = 0.35
FIG_SIZE_UNIT = 5
FIG_SIZE_2_by_4 = (FIG_SIZE_UNIT, FIG_SIZE_UNIT * 2)
FIG_SIZE_1_by_2 = (FIG_SIZE_UNIT / 2, FIG_SIZE_UNIT)
BOUND_SUBSAMPLE_RATE = 4

SKIP = 200

import random
random.seed(1234) 

def get_preds_file(tf, train_species, test_species, model_type):
    preds_root = ROOT + "/model_out/"
    return preds_root + tf + "_" + model_type + "_" + train_species + "-trained_" + test_species + "-test.preds.npy"

def load_all_test_set_preds(test_species='hg38'):
    # takes a while to run.
    preds_dict = defaultdict(lambda : dict())

    # loop over mouse-trained, human-trained models, and DA mouse-trained models
    for tf in TFS:
        if tf == "CEBPA":
            all_species = SPECIES
        else:
            all_species = SPECIES_SMALL
        models = big_include
        for model in models:
            
            print("=== " + tf + ", " + train_species + "-trained ===")

            # load predictions for model runs
            if "DA" in model:
                model_type = "kelly"
                train_species = model[:-3]
            else:
                model_type = "basic_model"
                train_species = model
            preds_file = get_preds_file(tf, train_species, test_species, model_type)
            try:
                preds_dict[model][tf] = np.load(preds_file).flatten()
            except:
                print("Could not load preds file:", preds_file)
            
    return preds_dict



def get_test_bed_file(tf, species):
    # This function returns the path to a BED-format file
    # containing the chromosome names, starts, ends, and
    # binding labels for all examples to test the model with.
    # This file is specific to each tf -- the last column
    # should contain the binding label for each window
    if species not in SPECIES:
        species = test_species
    return(ROOT + "exp_data/" + species + "/" + tf + "/chr2.bed")


def get_test_labels(tf, species="hg38"):
    # This function reads in the test-data bed file 
    # for a given species and TF and returns the binding labels
    # for each example in that file.
    
    labels_file = get_test_bed_file(tf, species)
    with open(labels_file) as f:
        return np.array([int(line.split()[-1]) for line in f])

def get_test_labels_all_tfs(species="hg38"):
    labels = dict()
    for tf in TFS:
        labels_for_tf = get_test_labels(tf, species)
        len_to_truncate_by = avg_preds["hg38"][tf].shape[0]
        # for species in avg_preds:
        #   if len_to_truncate_by > avg_preds[species][tf].shape[0]:
        #       len_to_truncate_by = avg_preds[species][tf].shape[0]
        labels[tf] = labels_for_tf[:len_to_truncate_by]
        # labels[tf] = labels_for_tf
    return labels

def make_preds_and_labels_dfs(avg_preds, labels):
    preds_dfs = dict()
    for tf in TFS:
        dict_to_make_into_df = {"labels" : labels[tf]}
        goal_len = labels[tf].shape[0]
        
        for model in big_include:
            try:
                preds_from_train_species = avg_preds[model][tf]
                # assert preds_from_train_species.shape[0] == goal_len
                dict_to_make_into_df[train_species] = preds_from_train_species
            except:
                pass
            
        preds_dfs[tf] = pd.DataFrame(dict_to_make_into_df)
        
    return preds_dfs

def get_model_name(model, test_species):
    if model in SPECIES:
        model_name = model_names_dict[model] + "-trained"
    else:
        spec = model[:-3]
        model_name = "DA: " + model_names_dict[spec] + "-trained"
    return model_name
        

def bound_scatterplot(model1_preds, model2_preds, tf_name,
                      plot_index, model_names):
    # This function draws a single scatterplot of bound sites (subplot of figure).
    # model1_preds: x-axis values for all points to plot
    # model2_preds: y-axis values for all points to plot
    # plot_index: either 0 or 1. 0 = top plot in column, 1 = bottom plot.
    # model_names: plot-acceptable names for the models that generated the x-axis
    #     and y-axis predictions, respectively. Expecting a list of length 2.
    
    # First, a random sample of sites are chosen, so that
    # the plot is not too overcrowded
    model_preds_subsample = random.sample(list(zip(model1_preds, model2_preds)),
                            k = int(len(model1_preds) / BOUND_SUBSAMPLE_RATE))
    model1_preds_subsample = [pair[0] for pair in model_preds_subsample]
    model2_preds_subsample = [pair[1] for pair in model_preds_subsample]
    
    # Then each bound site is plotted as an individual dot on a scatter plot
    plt.scatter(model1_preds_subsample, model2_preds_subsample,
                alpha = ALPHA, s = DOT_SIZE, c = "#007DEA")
    
    # adjust axes to show all points, add ticks
    plt.xlim(0 - AX_OFFSET, 1 + AX_OFFSET)
    plt.ylim(0 - AX_OFFSET, 1 + AX_OFFSET)
    plt.xticks([0, 0.5, 1])
    plt.yticks([0, 0.5, 1])
    
    # add axis labels
    if len(model_names[1]) > 5:
        plt.ylabel(model_names[1] + "\nModel Prediction", fontsize = AXIS_SIZE)
    else:
        plt.ylabel(model_names[1] + " Model Prediction", fontsize = AXIS_SIZE)
        
    # add x-axis label only if this subplot is the bottom row of the figure
    if plot_index == len(TFS) - 1:
        if len(model_names[0]) > 5:  # adjust fontsize for longer model names
            plt.xlabel(model_names[0] + "\nModel Prediction", fontsize = AXIS_SIZE)
        else:
            plt.xlabel(model_names[0] + " Model Prediction", fontsize = AXIS_SIZE)
        
    # add second "axis" to write TF name to the left of the plot
    # only do this for bound scatterplots because they are in left column of figure
    ax2 = plt.gca().twinx()
    if len(model_names[1]) > 5:
        ax2.spines["left"].set_position(("axes", 0 - 1.2 * TF_TWINAX_OFFSET))
    else:
        ax2.spines["left"].set_position(("axes", 0 - TF_TWINAX_OFFSET))
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('none')
    ax2.set_yticklabels([])
    ax2.set_ylabel(tf_name, fontsize = AXIS_SIZE + 2)
    
    # add text above subplot only if we are drawing in the top row of the figure
    if plot_index == 0:
        ax3 = plt.gca().twiny()
        ax3.spines["top"].set_position(("axes", 1))
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        ax3.set_xlabel("Bound Sites", fontsize = AXIS_SIZE + 2)
    
    
    
def unbound_scatterplot(model1_preds, model2_preds,
                        plot_index, model_names):
    # This function draws a single scatterplot of unbound sites.
    # model1_preds: x-axis values for all points to plot
    # model2_preds: y-axis values for all points to plot
    # plot_index: either 0 or 1. 0 = top plot in column, 1 = bottom plot.
    # model_names: plot-acceptable names for the models that generated the x-axis
    #     and y-axis predictions, respectively. Expecting a list of length 2.
    
    # no subsampling here, as in bound_scatterplot(),
    # because we already subsampled unbound sites using SKIP
    plt.scatter(model1_preds, model2_preds, alpha = ALPHA, s = DOT_SIZE, c = "#D60242")
    
    # adjust axes
    plt.xlim(0 - AX_OFFSET, 1 + AX_OFFSET)
    plt.ylim(0 - AX_OFFSET, 1 + AX_OFFSET)
    plt.xticks([0, 0.5, 1])
    
    # label x-axis only if we are drawing subplot in bottom row of figure
    if plot_index == len(TFS) - 1:
        if len(model_names[0]) > 5:  # adjust fontsize for longer model names
            plt.xlabel(model_names[0] + "\nModel Prediction", fontsize = AXIS_SIZE)
        else:
            plt.xlabel(model_names[0] + " Model Prediction", fontsize = AXIS_SIZE)
        
    # add text above subplot only if we are drawing in the top row of the figure
    if plot_index == 0:
        ax2 = plt.gca().twiny()
        ax2.spines["top"].set_position(("axes", 1))
        ax2.set_xticklabels([])
        ax2.set_xticks([])
        ax2.set_xlabel("Unbound Sites", fontsize = AXIS_SIZE + 2)

        
        
def generate_bound_unbound_scatters(preds_and_labels_dfs, train_species,
                                    save_files = False):
    # This function generates the full Figure 4,7, or 10 (bound and unbound sites).
    # preds_dict: a 3-layer dictionary, where keys for layer 1 are ["bound", "unbound"],
    #     keys for layer 2 are TF names, and keys for layer 3 are model type / species
    #     names (["mm10", "DA", "hg38"]).
    # train_species: a list of length 2 containing the model type / species names for
    #     the model predictions to plot on the x and y axes, respectively. Will be used
    #     to index into layer 3 of preds_dict.
    
    assert len(train_species) == 2, train_species
    
    # translate short-hand model type names for plot-acceptable names
    if test_species not in SPECIES:
        test_tmp = 'hg38'
    else:
        test_tmp = test_species
    model_names = [get_model_name(string, test_tmp) for string in train_species]

    # setup subplots: two columns (1 for bound sites, 1 for unbound, 4 rows (1 per TF)
    mpl.rcParams.update(mpl.rcParamsDefault)
    fig, ax = plt.subplots(nrows = len(TFS), ncols = 2, figsize = FIG_SIZE_2_by_4,
                           sharex = True, sharey = True,
                           gridspec_kw = {'hspace': 0.08, 'wspace': 0.3})

    # iterate over rows of subplots
    for plot_index,tf in enumerate(TFS):
        # left subplot in this row will be for bound sites
        plt.sca(ax[plot_index][0])
        
        bound_sites_for_tf = preds_and_labels_dfs[tf]
        bound_sites_for_tf = bound_sites_for_tf[bound_sites_for_tf["labels"] == 1]
        
        bound_scatterplot(bound_sites_for_tf[train_species[0]],
                          bound_sites_for_tf[train_species[1]],
                          TFS[plot_index], plot_index, model_names)

        # right subplot in this row will be for unbound sites
        plt.sca(ax[plot_index][1])
        
        unbound_sites_for_tf = preds_and_labels_dfs[tf]
        unbound_sites_for_tf = unbound_sites_for_tf[unbound_sites_for_tf["labels"] == 0]
        
        if SKIP is not None:
            unbound_sites_for_tf = unbound_sites_for_tf[::SKIP]
        
        unbound_scatterplot(unbound_sites_for_tf[train_species[0]],
                            unbound_sites_for_tf[train_species[1]],
                            plot_index, model_names)
    
    if save_files:
        # plt.savefig(ROOT + "plots/scatter_" + train_species[0] + "_" + train_species[1] + ".pdf",
        #             bbox_inches='tight', pad_inches = 0)
        plt.savefig(ROOT + "plots/scatter_" + train_species[0] + "_" + train_species[1] + ".png",
                    bbox_inches='tight', pad_inches = 0)
        print("DONE!")

SAVE_FILES = True

if test_species not in SPECIES:
    test_tmp = 'hg38'
else:
    test_tmp = test_species

avg_preds = load_all_test_set_preds(test_tmp)

labels = get_test_labels_all_tfs(species=test_tmp)
preds_and_labels_dfs = make_preds_and_labels_dfs(avg_preds, labels)

generate_bound_unbound_scatters(preds_and_labels_dfs,
                                train_species = [train_species, test_species],
                                save_files = SAVE_FILES)

