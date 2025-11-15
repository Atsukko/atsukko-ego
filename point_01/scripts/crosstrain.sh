#!/bin/bash

# Set default values
DATA_DIR=""     # Data directory (required)
CLUSTER=""      # Cluster name (required if sbatch is available)
TIME="06:00:00"   # Default time is 6 hours

# Function to display usage information
usage() {
  echo "Usage: $0 [-d <data_dir>] [-c <cluster>] [-t <time>]"
  echo "  -d <data_dir>:  Path to the data directory (required)."
  echo "  -c <cluster>:   Slurm cluster name (required only if sbatch is available)."
  echo "  -t <time>:      Slurm time limit (HH:MM:SS, default: 06:00:00)."
  echo "  -h:             Display this help message."
  exit 1
}

# Parse command-line options using getopts
while getopts "d:c:t:h" opt; do
  case "$opt" in
    d)
      DATA_DIR="$OPTARG"
      ;;
    c)
      CLUSTER="$OPTARG"
      ;;
    t)
      TIME="$OPTARG"
      ;;
    h)
      usage
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      usage
      ;;
  esac
done

# Check if the required DATA_DIR is provided
if [ -z "$DATA_DIR" ]; then
  echo "Error: Data directory (-d) is required."
  usage
fi

# Determine the root directory based on the script's location
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Construct the path to the training script
TRAIN_SCRIPT="$ROOT_DIR/scripts/train.py"
CACHE_SCRIPT="$ROOT_DIR/scripts/cache.py"

# Assert that the training script exists
if [ ! -f "$TRAIN_SCRIPT" ]; then
  echo "Error: Training script not found at $TRAIN_SCRIPT."
  exit 1
fi

# Function to check if sbatch is available
is_sbatch_available() {
  if command -v sbatch >/dev/null 2>&1; then
    return 0  # sbatch is available
  else
    return 1  # sbatch is not available
  fi
}

# Define the slurm submission script path.  Assume it's in the same directory.
RUN_ON_SLURM="$SCRIPT_DIR/run_on_slurm.sh"

# Check if sbatch is available
is_sbatch_available
SBATCH_AVAILABLE=$?

# If cluster is provided but sbatch is not available, raise an error
if [[ ! -z "$CLUSTER" && $SBATCH_AVAILABLE -ne 0 ]]; then
  echo "Error: Cluster specified (-c) but sbatch is not available."
  usage
fi

if [[ $SBATCH_AVAILABLE -eq 0 ]]; then
  echo "sbatch is available. Submitting jobs to Slurm."
  # Check if cluster is provided
  if [ -z "$CLUSTER" ]; then
    echo "Error: Cluster name (-c) is required when submitting jobs to Slurm."
    usage
  fi

  # Training commands with variables, now submitted via Slurm
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part1 --override experiment-name=crosstraining-backbone-part1 --override training.learning_rate=1e-5
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part2 --override experiment-name=crosstraining-backbone-part2 --override training.learning_rate=1e-5
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part3 --override experiment-name=crosstraining-backbone-part3 --override training.learning_rate=1e-5
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part4 --override experiment-name=crosstraining-backbone-part4 --override training.learning_rate=1e-5
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part5 --override experiment-name=crosstraining-backbone-part5 --override training.learning_rate=1e-5
  $RUN_ON_SLURM --cluster "$CLUSTER" --time "$TIME" -- python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part6 --override experiment-name=crosstraining-backbone-part6 --override training.learning_rate=1e-5
else
  echo "sbatch is not available. Running jobs sequentially"

  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part1 --override experiment-name=crosstraining-backbone-part1 --override training.learning_rate=1e-5
  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part2 --override experiment-name=crosstraining-backbone-part2 --override training.learning_rate=1e-5
  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part3 --override experiment-name=crosstraining-backbone-part3 --override training.learning_rate=1e-5
  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part4 --override experiment-name=crosstraining-backbone-part4 --override training.learning_rate=1e-5
  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part5 --override experiment-name=crosstraining-backbone-part5 --override training.learning_rate=1e-5
  python "$TRAIN_SCRIPT" --data "$DATA_DIR" --experiment backbone --override dataset=cross_training/part6 --override experiment-name=crosstraining-backbone-part6 --override training.learning_rate=1e-5

fi

echo "Training completed."

