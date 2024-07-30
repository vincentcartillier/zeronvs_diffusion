#!/bin/bash


# -- helper
echo -e "## Running ZeroNVS overfiting experiments on EgoExo \n"

# Function to display the command help
show_help() {
    echo    "## Usage: $0 [options]"
    echo -e "          $0 <output_dir> <name> <configs> \n"
    echo
    echo "Options:"
    echo "  --help     Show this help message and exit"
    echo
}

# Check for the --help flag
if [[ -z "$1" || -z "$2" || -z "$3" || "$1" == "-h" || "$1" == "--help" || "$1" == "--helper" ]]; then
    show_help
    exit 0
fi


# -- set variables
ROOT="/srv/essa-lab/flash3/vcartillier3/zeronvs_diffusion/zero123/"

SCRIPT_DIR="$ROOT/batch_scripts"

RUNNER_WRAPPER_SCRIPT="$SCRIPT_DIR/template_wrapper.sh"

UNIQUE_ID=$((1 + RANDOM % 100000))

exclude_nodes="voltron"
echo -e "## excluding these nodes: $exclude_nodes \n"

logdir="$1"
expename="$2"
config_file="$3"

echo "config path: $config_file"

# create copy of runner_wrapper.sh template
tmp_wrapper_name="wrapper_tmp_${UNIQUE_ID}.sh"
dst_wrapper="$SCRIPT_DIR/tmp/$tmp_wrapper_name"

cp "$RUNNER_WRAPPER_SCRIPT" "$dst_wrapper"

chmod +x "$dst_wrapper"

sbatch --exclude="$exclude_nodes" slurm_script.sh "$dst_wrapper" "$logdir" "$expename" "$config_file"



