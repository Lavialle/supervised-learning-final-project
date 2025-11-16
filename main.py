"""
Main script to select and run different machine learning models.
Models available:
- Individual model training and evaluation
- Hyperopt optimization for CatBoost and XGBoost models
- Stacking model with Hyperopt optimization
"""

import argparse
import subprocess
import sys
import os

from models.individual_model import run_individual
from models.hyperopt_model import run_hyperopt
from models.stacking_model import run_stacking

import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

def install_requirements():
    """Install packages from requirements.txt if necessary."""
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        try:
            print(f"Package installation from {requirements_file}...")
            subprocess.check_call(["uv", "pip", "install", "-r", requirements_file])
        except subprocess.CalledProcessError:
            print("Erreur lors de l'installation des packages.")
            exit(1)
    else:
        print(f"Le fichier {requirements_file} est introuvable.")
        exit(1)


def main():
    install_requirements()
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Select a model to run.")
    parser.add_argument(
        "--model",
        type=str,
        choices=["individual", "hyperopt", "stacking"],
        required=True,
        help="Choose the model to run: individual, hyperopt, stacking"
    )

    if len(sys.argv) == 1:  # No arguments passed
        parser.print_help()
        exit(1)

    args = parser.parse_args()

    # Call the selected model
    if args.model == "individual":
        run_individual()
    elif args.model == "hyperopt":
        run_hyperopt()
    elif args.model == "stacking":
        run_stacking()

if __name__ == "__main__":
    main()