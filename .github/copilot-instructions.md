# Copilot Instructions for Energy-efficiency

## Project Overview
This repository contains Python scripts and modules for energy efficiency analysis, modeling, and prediction. The codebase includes machine learning models (CNN-LSTM, SVR), command-line utilities, and packaging scripts. The structure supports both direct script execution and packaging for distribution.

## Architecture & Major Components
- **Top-level Scripts**: `cnn-lstm.py`, `svr.py`, `Program.py`, `Program2.py`, `try.py` — entry points for different modeling and analysis workflows.
- **Energy/ Subdirectory**: Contains mirrored scripts and packaging files for modular development and distribution.
- **application/**: Likely for application-specific logic (details may need further exploration).
- **build/**: Contains build artifacts, especially for packaged executables (e.g., via PyInstaller).

## Developer Workflows
- **Build**: Use the VS Code build task (`msbuild`) for Windows-specific packaging. Artifacts are placed in `build/command/`.
- **Python Packaging**: Use `setup.py` and `requirements.txt` in both root and `Energy/` for dependency management and distribution.
- **Spec Files**: `Energy.spec` and `Energy/command.py` are used for PyInstaller builds.

## Patterns & Conventions
- Scripts are duplicated in both root and `Energy/` for modularity and packaging.
- Build outputs and logs are stored in `build/command/` and `Energy/build/command/`.
- Use `.spec` files for custom build configurations.
- External dependencies are managed via `requirements.txt`.
- Avoid editing files in `build/` directly; regenerate via build workflows.

## Integration Points
- **PyInstaller**: Used for creating Windows executables; see `.spec` files and build outputs.
- **MSBuild**: Invoked via VS Code tasks for Windows builds.
- **Python**: All modeling and analysis logic is in Python; ensure environment matches `requirements.txt`.

## Examples
- To add a new model, create a script in the root and mirror it in `Energy/` if packaging is needed.
- To update dependencies, edit `requirements.txt` in both locations and rebuild.
- To debug build issues, check logs in `build/command/warn-command.txt`.

## Key Files & Directories
- `cnn-lstm.py`, `svr.py`, `Program.py`, `try.py`
- `Energy/command.py`, `Energy.spec`, `Energy/setup.py`, `Energy/requirements.txt`
- `build/command/`, `Energy/build/command/`

---

**If any section is unclear or missing, please provide feedback or specify which workflows or patterns need more detail.**
