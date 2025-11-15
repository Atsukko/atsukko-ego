#!/bin/bash

# Function to display usage information
usage() {
  echo "Usage: $0 [--name <job_name>] [--cluster <cluster_name>] [--time <HH:MM:SS>] [--] <script>"
  echo
  echo "Options:"
  echo "  --name      Name of the job (used for SBATCH job-name). Default is 'default_job'."
  echo "  --cluster   GPU cluster to use. If not specified, defaults based on script logic."
  echo "  --time      Total SBATCH time (format: HH:MM:SS). Default is '11:59:00'."
  echo "  --help      Display this help message."
  echo
  echo "The script to execute should be provided after the '--' argument."
  exit 1
}

# Raise an error if sbatch is not available
if ! command -v sbatch &>/dev/null; then
  echo "Error: sbatch command not found. Please run this script on a SLURM cluster."
  exit 1
fi

# Default values
JOB_NAME="default_job"
PARTITION=""
SBATCH_TIME="11:59:00"

# Parse command-line arguments using getopt
PARSED_ARGS=$(getopt -o h --long name:,cluster:,time:,help -- "$@")
if [[ $? -ne 0 ]]; then
  usage
fi

eval set -- "$PARSED_ARGS"

# Process the parsed arguments
while true; do
  case "$1" in
    --name)
      JOB_NAME="$2"
      shift 2
      ;;
    --cluster)
      PARTITION="$2"
      shift 2
      ;;
    --time)
      SBATCH_TIME="$2"
      shift 2
      ;;
    --help|-h)
      usage
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# Check if a script is provided after '--'
if [[ $# -eq 0 ]]; then
  echo "Error: No script provided after '--'."
  usage
fi

# Check that a partition is specified
if [[ -z "$PARTITION" ]]; then
  echo "Error: No partition specified. Use --cluster to specify a GPU cluster."
  usage
fi

# The remaining arguments after '--' are the script to execute
SCRIPT_TO_EXECUTE="$@"

# Set the parent directory based on the script's location
PARENT_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATETIME=$(date '+%Y-%m-%d_%H-%M-%S.%3N')

# Setting up the output folders
mkdir -p "$PARENT_DIR/slurm_logs"
OUTPUT_FILE="$PARENT_DIR/slurm_logs/${DATETIME}.txt"

echo "Follow with 'tail -f $OUTPUT_FILE'"

# Temporary script with sbatch directives
TEMP_SCRIPT=$(mktemp /tmp/temp_script.XXXXXX)

# Write sbatch directives to the temporary script
cat <<EOT >"$TEMP_SCRIPT"
#!/bin/bash
#SBATCH -p $PARTITION
#SBATCH -t $SBATCH_TIME
#SBATCH -o $OUTPUT_FILE
#SBATCH -e $OUTPUT_FILE
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH --gres gpu:1
#SBATCH --job-name=${JOB_NAME}_${DATETIME}
#SBATCH --signal=SIGUSR1@90

$(echo "$SCRIPT_TO_EXECUTE")
EOT

# Make the temporary script executable
chmod +x "$TEMP_SCRIPT"

# Submit the job
sbatch "$TEMP_SCRIPT"

# Remove the temporary script
rm "$TEMP_SCRIPT"

