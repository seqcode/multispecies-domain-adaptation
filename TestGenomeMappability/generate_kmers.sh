#PBS -W umask=0007
#PBS -W group_list=sam77_collab
#PBS -l walltime=50:00:00
#PBS -l nodes=1:ppn=8
#PBS -j oe
#PBS -A sam77_h_g_sc_default
#PBS -l mem=100gb

genome=rn5

cd /storage/home/vza5092/group/projects/cross-species/liver-tfs/scripts

gInfo=/storage/home/vza5092/group/projects/cross-species/liver-tfs/exp_data/$genome/info/$genome.info
genomeDir=/storage/home/vza5092/group/genomes/$genome/

python3 generate_kmers.py $gInfo $genomeDir 36
