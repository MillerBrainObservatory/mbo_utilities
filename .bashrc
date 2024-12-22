# Enable the subsequent settings only in interactive sessions
case $- in
  *i*) ;;
    *) return;;
esac


export SSH_KEY_PATH="~/.ssh/rsa_id"

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/ru-auth/local/home/mbo_soft/miniforge3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ru-auth/local/home/mbo_soft/miniforge3/etc/profile.d/conda.sh" ]; then
        . "/ru-auth/local/home/mbo_soft/miniforge3/etc/profile.d/conda.sh"
    else
        export PATH="/ru-auth/local/home/mbo_soft/miniforge3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

alias gologs='vim ~/logs/'
alias runsb='sbatch ~/repos/utilities/slurm/multifile_batch.slurm'

export SCRATCH="/lustre/fs4/mbo/scratch/"
export MBO_DATA="${SCRATCH}/mbo_data/"
export MBO_SOFT="${SCRATCH}/mbo_soft/"
export SCRATCH_USER="${SCRATCH}/mbo_soft/"
export USER_DATA="${SCRATCH}/${USER}/data/"

SPACK_RELEASE=spack_2020b
source /ru-auth/local/home/ruitsoft/soft/spackrc/spackrc.sh

alias nvim="${MBO_SOFT}/bin/nvim"
alias soft="${MBO_SOFT}"
alias data="${MBO_DATA}"
alias squ='squeue -u $USER'
alias gor='cd ~/repos'
alias gob='nvim ~/.bashrc'
