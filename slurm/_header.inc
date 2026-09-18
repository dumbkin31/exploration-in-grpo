# Common #SBATCH header for Ada (nlp account, 2080 Ti nodes). Copy into each script; SLURM
# does not support includes. Values mirror configs/ada.env.sh (MC_SLURM_*).
#SBATCH -A nlp
#SBATCH --qos=normal
#SBATCH -p u22
#SBATCH -C 2080ti
#SBATCH -c 40
#SBATCH --mem-per-cpu=3G
#SBATCH --signal=B:SIGUSR1@300
#SBATCH -o slurm-%x-%j.out
