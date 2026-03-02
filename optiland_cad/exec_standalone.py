"""
Standalone execution script for Optiland CAD conversion.
Modify the variables in the CONFIGURATION section and run this file directly.
"""
from optiland_cad.main import main  # Ensure optiland_cad is installed in your environment

# ==========================================
# CONFIGURATION - Set your parameters here
# ==========================================

# 1. Choose your INPUT TYPE (Fill only ONE of these, leave others as None)
INPUT_JSON = "input.json"       # Path to an existing Optiland JSON
SAMPLE_NAME = None              # e.g., "Edmund_49_847"
ZEMAX_FILE = None               # Path to a .zmx file

# 2. Output Settings
OUTPUT_DIR = "step_output"      # Where to save the STEP files
FILE_PREFIX = "my_optic"        # Prefix for the generated filenames

# 3. Extra Options
JSON_ONLY = False               # If True, only generates JSON, skips STEP export
SAVE_JSON_PATH = "output.json"  # Path to save JSON (required for Samples/Zemax)
SURFACE_ONLY = False            # Export only the surface_group portion

# ==========================================
# EXECUTION LOGIC
# ==========================================

def run_conversion():
    # Build the argument list dynamically based on your settings
    argv = []

    # Handle Input
    if INPUT_JSON:
        argv.append(INPUT_JSON)
    elif SAMPLE_NAME:
        argv.extend(["--sample", SAMPLE_NAME])
    elif ZEMAX_FILE:
        argv.extend(["--zemax", ZEMAX_FILE])
    else:
        print("Error: Please provide an INPUT_JSON, SAMPLE_NAME, or ZEMAX_FILE.")
        return

    # Handle Outputs & Flags
    argv.extend(["--out", OUTPUT_DIR])
    argv.extend(["--prefix", FILE_PREFIX])

    if SAVE_JSON_PATH:
        argv.extend(["--json-out", SAVE_JSON_PATH])
    
    if JSON_ONLY:
        argv.append("--json-only")
    
    if SURFACE_ONLY:
        argv.append("--surface-group-only")

    # Execute the original logic
    print(f"Starting conversion with arguments: {' '.join(argv)}")
    main(argv)
    print("\nProcess finished.")

if __name__ == "__main__":
    run_conversion()