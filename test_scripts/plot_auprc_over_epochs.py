import numpy as np
import matplotlib.pyplot as plt
from math import ceil


ROOT = "/storage/home/vza5092/group/projects/cross-species/liver-tfs/"

SPECIES = ["mm10", "hg38", "canFam4", "rn5", "rheMac10", "monDom5", "galGal6"]
SPECIES_SMALL = ["mm10", "hg38", "canFam4", "rn5", "rheMac10"]

model_names_dict = {"mm10" : "Mouse", "hg38" : "Human", "monDom5" : "Possum", "canFam4" : "Dog", "galGal6" : "Chicken", "rn5" : "Rat", "rheMac10" : "Macaque"}

colors_dict = {"mm10" : "tab:blue", "hg38" : "tab:orange", "monDom5" : "tab:green", "canFam4" : "tab:red", "galGal6" : "tab:purple", "rn5" : "tab:brown", "rheMac10" : "tab:pink"}

TFS = ["CEBPA", "FoxA1", "HNF4a", "HNF6"]

DOT_SIZE = 5
ALPHA = 0.5
AXIS_SIZE = 11
AX_OFFSET = 0.02
TF_TWINAX_OFFSET = 0.35
FIG_SIZE_UNIT = 2.5
# FIG_SIZE_2_by_4 = (FIG_SIZE_UNIT, FIG_SIZE_UNIT * 2)
# FIG_SIZE_1_by_4 = (FIG_SIZE_UNIT / 2, FIG_SIZE_UNIT * 2)

def get_model_log_files(species, tf, model_type, runs = 5):
    # This function supplies a list of log filenames.
    # Each filename is for the log file for a specific run/replicate.
    # See the scripts run_training.sh and run_DA_training.sh for
    # log file creation (output of model training). See the function
    # get_both_species_auprcs() below for expected log file content.
    log_out_root = ROOT + "logs/training/"
    
    prefix = log_out_root + model_type + "_" + species + "_" + tf + "_run"
    suffix = ".log"
    
    return [prefix + str(i) + suffix for i in range(1, runs + 1)]

def get_all_species_auprcs(model_out_filename):
    # This function reads in the info stored in a single log file.
    # The log files contain a lot of info/junk, but we only care
    # about the auPRCs for each epoch. Those can be found on lines
    # that start with "auPRC:" and are listed in the file in order
    # of epoch. Each epoch, both the source species' and target
    # species' auPRCs are listed, with the target species' auPRC
    # listed first.
    # This function returns a tuple of two lists: list 1 is the
    # auPRCs across each epoch when the model was evaluated on
    # mouse data; list 2 is the auPRCs across each epoch when the
    # model was evaluated on human data.
    
    lines = {}
    if "CEBPA" in model_out_filename:
        for spec in SPECIES:
            lines[spec] = []
    else:
        for spec in SPECIES_SMALL:
            lines[spec] = []

    species = "hg38"
    try:
        with open(model_out_filename) as f:
            # assuming auPRCs are listed by epoch
            # with target species listed first, then source species
            for line in f:
                if line.startswith("==== "):
                    species = line.strip().split()[-2]
                if line.startswith("auPRC"):
                    auprc = float(line.strip().replace("auPRC:\t", ""))
                    lines[species].append(auprc)
    except Exception as e:
        print(e)
               
    return lines

def auprc_lineplot(model_files, tf_name, plot_index, plot_y_index,
                   y_max = None, title = None):
    # This function creates a single line subplot (to be called repeatedly).
    # Arguments:
    #     - model_files: paths for the log files for all model runs,
    #           for a given TF (output of get_model_log_files())
    #     - tf_name: name of the TF to display on the plot
    #     - plot_index: the top-to-bottom index of the subplot
    #     - plot_y_index: the left-to-right index of the subplot
    #     - y_max: optional, manually set the top limit of the y-axis
    #           for this subplot (default auto-detects max of data plotted)
    #     - title: optional
    ax = plt.gca()
    
    # First, load in the auPRCs across all epochs for all model runs
    # Keep track of the max auPRC so the y-axis limits can be set properly
    # Also keep track of the legend handles to use later
    max_auprc_so_far = 0
    legend_handles = []

    if tf_name == "CEBPA":
        all_species = SPECIES
    else:
        all_species = SPECIES_SMALL

    max_auprc_so_far = 0
    specs = list(model_files.keys())
    lines = []
    for spec in specs:
        for model_out_file in model_files[spec]:
            auprcs = get_all_species_auprcs(model_out_file)
            specs2 = list(auprcs.keys())
            spec2 = specs2[plot_y_index]
            if max_auprc_so_far < max([max_auprc_so_far] + auprcs[spec2]):
                max_auprc_so_far = max([max_auprc_so_far] + auprcs[spec2])
            lines.append(ax.plot(range(len(auprcs[spec2])), auprcs[spec2],
                                c = colors_dict[spec], alpha=ALPHA)[0])
        
    legend_handles = lines

    # set top limit of y-axis
    if y_max is None:
        y_max = max_auprc_so_far
    ax.set_ylim( - y_max / 25, y_max + 0.02)
    
    if y_max > 0.4:
        possible_ticks = [num / 10 for num in list(range(0, 11, 2))]
        ticks_to_use = [num for num in possible_ticks if num < y_max]
    else:
        possible_ticks = [num / 10 for num in list(range(0, 11, 1))]
        ticks_to_use = [num for num in possible_ticks if num < y_max]
    
    ax.set_yticks(ticks_to_use)
    ax.set_yticklabels(ticks_to_use)
    
    # if we are plotting a subplot in the leftmost column...
    if plot_y_index == 0:
        # label the y-axis with "auPRC"
        ax.set_ylabel("auPRC", fontsize = AXIS_SIZE)
        
        # add the TF name label to the far left of the plot
        ax2 = plt.gca().twinx()
        ax2.spines["left"].set_position(("axes", 0 - TF_TWINAX_OFFSET))
        ax2.yaxis.set_label_position('left')
        ax2.yaxis.set_ticks_position('none')
        ax2.set_yticklabels([])
        ax2.set_ylabel(tf_name, fontsize = AXIS_SIZE + 2)
    else:
        ax.set_yticklabels([])
        

    
    # if we're drawing a subplot in the top row of the plot...
    if plot_index == 0:
        # draw an invisible extra axis on top of the subplot
        ax3 = plt.gca().twiny()
        ax3.spines["top"].set_position(("axes", 1))
        ax3.set_xticklabels([])
        ax3.set_xticks([])
        
        # if we're drawing a subplot in the left column...
        if title is None:
            ax3.set_xlabel(model_names_dict[spec2] + " Validation Set", fontsize = AXIS_SIZE + 1)
        else:
            ax3.set_xlabel(title, fontsize = AXIS_SIZE + 1)
        
    max_plots = 3
    if spec2 not in SPECIES_SMALL:
        max_plots = 0
    # if you're drawing a subplot in the bottom row of the plot...
    if plot_index == max_plots:
        # add an x-axis for epochs
        ax.set_xlabel("Epochs", fontsize = AXIS_SIZE)
        ax.set_xticks([0, 5, 10, 15])
        ax.set_xticklabels([0, 5, 10, 15])
    else:
        # otherwise don't label the x-axis
        ax.set_xticks([])

    return legend_handles
        
        
def get_y_max(list_of_file_lists):
    # To ensure that the y-axis is the same scale across
    # a row of subplots, calculate the max limit in advance.
    # This max is calculated over all model log files to be
    # used in plotting (one for each replicate run).
    y_max = 0
    for file_list in list_of_file_lists:
        for model_out_file in file_list:
            auprcs = get_all_species_auprcs(model_out_file)
            auprc_list = []
            for spec in auprcs:
                auprc_list += auprcs[spec]
            y_max = max([y_max] + auprc_list)
    return y_max
    

def generate_all_auprc_plots(tf_list, save_file = False):
    # This function draws Figure 2.
    
    
    plt.rcParams.update(plt.rcParamsDefault)

    fig, ax = plt.subplots(nrows = len(tf_list), ncols = len(SPECIES), figsize = (FIG_SIZE_UNIT*len(SPECIES), FIG_SIZE_UNIT*len(TFS)),
                           gridspec_kw = {'hspace': 0.08, 'wspace': 0.08})
    for i in range(1, 4):
        ax[i,5].set_axis_off()
        ax[i,6].set_axis_off()

    legend_handles = []
    for plot_index, tf in enumerate(tf_list):  # iterating over rows of subplots

        # For each TF and each species, retrieve the model log files

        if tf == "CEBPA":
            all_species = SPECIES
        else:
            all_species = SPECIES_SMALL

        trained_files = {}
        for spec in all_species:
            trained_files[spec] = {tf : get_model_log_files(spec, tf, 'BM', 1) for tf in tf_list}


        y_max = get_y_max([trained_files[spec][tf] for spec in all_species])
        
        # draw the left subplot in this row
        plt.sca(ax[plot_index][0])
        

        model_files = {}
        for spec in all_species:
            model_files[spec] = trained_files[spec][tf]

        if tf == "CEBPA":
            legend_handles = auprc_lineplot(model_files, tf,
                                            plot_index, 0, y_max = y_max)
        else:
            _ = auprc_lineplot(model_files, tf,
                            plot_index, 0, y_max = y_max)
        for i in range(1, len(all_species)):
            plt.sca(ax[plot_index][i])
            _ = auprc_lineplot(model_files, tf,
                           plot_index, i, y_max = y_max)
    
    # add a legend below all the subplots
    if len(legend_handles) > 0:
        fig.legend(legend_handles,
                   [model_names_dict[spec] + "-trained Models" for spec in SPECIES],
                  loc = "lower center", ncol = ceil(len(SPECIES)/2),
                  bbox_to_anchor=[0.5, 0])
    
    if save_file:
        plt.savefig(ROOT + "plots/auprc_over_epochs.pdf", bbox_inches = 'tight', pad_inches = 0)
        plt.savefig(ROOT + "plots/auprc_over_epochs.png", bbox_inches = 'tight', pad_inches = 0, dpi = 300)
        print("DONE!")

if __name__ == "__main__":

    generate_all_auprc_plots(TFS, save_file = True)