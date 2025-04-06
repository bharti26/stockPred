import subprocess
import sys
import os


def install_package(package):
    print(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except Exception as e:
        print(f"Failed to install {package}: {e}")


if __name__ == "__main__":
    # Print current Python version and path
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Try to upgrade pip first
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("Successfully upgraded pip")
    except Exception as e:
        print(f"Failed to upgrade pip: {e}")

    # List of packages to install
    packages = [
        "dash",
        "dash-bootstrap-components",
        "plotly",
        "pandas",
        "numpy"
    ]

    # Install each package
    for package in packages:
        install_package(package)

    # Check installed packages
    print("\nInstalled packages:")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "list"])
    except Exception as e:
        print(f"Failed to list packages: {e}")
