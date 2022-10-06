#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb

export LD_LIBRARY_PATH="/storage/home/vza5092/usr/lib64:$LD_LIBRARY_PATH"

module load anaconda3/2020.07

source activate ~/work/tf_env

cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts/setup_scripts

python3 setup_onehot.py $genome $tf
