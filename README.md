# multispecies-domain-adaptation

This repository is based on @kellycochran's https://github.com/seqcode/cross-species-domain-adaptation. In this case, we train neural networks to predict transcription factor binding in one test species (hg38) given data on transcription factor binding in several other training species (mm10, rn5, rheMac10, canFam4, monDom5, galGal6).

In `setup_scripts`, we first perform all data preprocessing. 

* FASTQ files are retrieved from ChIP-Seq experiments
* Alignment is performed using Bowtie to determine the positions of each sequence within the genome
* Peak calling is performed using MultiGPS
* Genomic windows (500bp long, 50bp apart) are created
* Blacklisted regions (those with artificially high ChIP-Seq signal) are removed
* Non-uniquely mappable regions of the genome are removed
* Remaining genomic windows are labelled as "bound" or "unbound" based on MultiGPS results.
* Data is split up into training, validation, and test datasets.


Instructions to run from MultiGPS onward (these scripts need modification to filter out "weak peaks"):

* Run MultiGPS for all combinations of TF and species by just running `~/group/projects/cross-species/liver-tfs/multigps/run_all.sh`
* Label genomic windows (blacklisted/non-umap have already been filtered out) by running `~/group/projects/cross-species/liver-tfs/setup_scripts/setup_ALL.sh`
* Create training data by running `~/group/projects/cross-species/liver-tfs/setup_scripts/all_setup_training_data.sh`


In `model_training`, we train several different types of models:

* standard model architecture trained on each individual species
* standard model architecture trained on all training species together
* domain-adaptive model architecture trained on each individual training species (test hg38)
* domain-adaptive model architecture trained on all training species together (multi-class classifier)
* binary adversarial model architecture trained on all training species together (test hg38)
* ensemble model of all training species using tensorflow (linear layer)
* ensemble model of all training species using scikit-learn (linear regression)

Instructions to train models:

* Train all models by running:
    * basic single-species model: `run_training_basic_all.sh`
    * multispecies non-DA model: `run_training_multi_all.sh`
    * multispecies DA model: `run_training_DA_all.sh`
    * Kelly's single-species DA models: `~/group/projects/cross-species/liver-tfs/kelly_model_training/run_DA_training_all.sh`
    * my single-species DA (currently throwing error): `run_training_DA_single_all.sh`
    * regression model: `run_training_reg_all.sh`

Finally, in `test_scripts`, we run all downstream analyses. We generate model predictions on the held-out test data (hg38 chromosome 1) and create various plots describing results and model training statistics.

Instructions:

* Run `run_generate_preds.sh` to generate predictions on test set for basic, multispecies non-DA, multispecies DA, and ensemble models
* Run `run_generate_individual_preds.sh` to generate predictions on test set for basic and Kelly DA models
* Run `plot_auprc_dot_plots.py` with argument `hg38` (this is the test species) to compare model performance (file can be modified to create different plots)
* Run `plot_site_distribution_scatter.py` with two arguments: second argument is `hg38`, the test species. First argument is multispecies model type to compare to hg38 performance, e.g. 'ensemble\_model\_reg' or 'multi_model'.
* Run `plot_individual_site_distribution_scatter.py` with two arguments: second argument is `hg38`, the test species. First argument is analogous to `plot_site_distribution_scatter.py` but this is used for single-species model comparison; first argument may be e.g. 'mm10' or 'mm10\_DA'.