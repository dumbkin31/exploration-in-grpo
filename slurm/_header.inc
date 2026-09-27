# Common #SBATCH header for Ada (research account / QoS low, 2080 Ti nodes: 1 GPU, 10 CPUs, 30 GB per
# job). Copy into each script; SLURM does not support includes. These are the research_1gpu defaults;
# `make sbatch-*` overrides them with $(mc_sbatch_args) from configs/ada.env.sh (MC_LAYOUT).
#SBATCH -A research
#SBATCH --qos=low
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -N 1
#SBATCH -c 10
#SBATCH --mem-per-cpu=3000M
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
