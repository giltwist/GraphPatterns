#!/bin/bash
#SBATCH -o output.%j
#SBATCH --time=20:00:00      # 20 hours of wall time
#SBATCH -c 4
#SBATCH -A csds433
#SBATCH --mem=120GB

module purge
module load Python/3.8.6-GCCcore-10.2.0
echo "Bringing up Python Virtual Environment"
cd ~/GraphPatterns

source maggievirt38/bin/activate
echo "Ensuring that all dependencies are ready...."
pip install -r requirements.txt

echo "Beginning the Graph Patterns Processing"
python3 -u graph_patterns.py > ~/GraphPatterns/output.txt

