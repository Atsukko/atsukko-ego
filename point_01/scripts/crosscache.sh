#!/bin/bash

# Fail if any command fails
set -e

# Set default values
DATA_DIR=""     # Data directory (required)

# Function to display usage information
usage() {
  echo "Usage: $0 [-d <data_dir>]"
  echo "  -d <data_dir>:  Path to the data directory (required)."
  echo "  -h:             Display this help message."
  exit 1
}

# Parse command-line options using getopts
while getopts "d:h" opt; do
  case "$opt" in
    d)
      DATA_DIR="$OPTARG"
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
CACHE_SCRIPT="$ROOT_DIR/scripts/cache.py"

python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part1 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part2 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part3 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part4 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part5 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load crosstraining-backbone-part6 --cache-name motions --only-val --out-key joints_3D_cc
python "$CACHE_SCRIPT" --data "$DATA_DIR" --load backbone --cache-name motions --only-val --out-key joints_3D_cc
