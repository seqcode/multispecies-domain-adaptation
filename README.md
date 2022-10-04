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

In `model_training`, we train several different types of models:

* standard model architecture trained on each individual species
* standard model architecture trained on all training species together
* domain-adaptive model architecture trained on each individual training species (test hg38)
* domain-adaptive model architecture trained on all training species together (multi-class classifier)
* binary adversarial model architecture trained on all training species together (test hg38)
* ensemble model of all training species using tensorflow (linear layer)
* ensemble model of all training species using scikit-learn (linear regression)

Finally, in `test_scripts`, we run all downstream analyses. We generate model predictions on the held-out test data (hg38 chromosome 1) and create various plots describing results and model training statistics.