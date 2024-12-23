#!/bin/bash -l

#SBATCH --job-name=gsearch
#SBATCH --output=logs/grid_search_%x_%j.out
#SBATCH --error=logs/grid_search_%x_%j.err
#SBATCH --time=9:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=34
#SBATCH --mem=0
#SBATCH --partition=hpc_a10_a

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: sbatch $0 <input_file>"
  exit 1
fi

INPUT_FILE=$1
BASE_DIR=$(dirname "$INPUT_FILE")
FILE_NAME=$(basename "$INPUT_FILE")
GRID_SEARCH_DIR="$BASE_DIR/grid_search"
BATCH_PATH="$GRID_SEARCH_DIR/batch.pickle"

# Create grid_search directory if it doesn't exist
mkdir -p "$GRID_SEARCH_DIR"

lcp --batch_path "$BATCH_PATH" --data_path "$INPUT_FILE" --run mcorr

