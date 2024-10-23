{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    # Python environment
    python312
    python312Packages.pip
    python312Packages.setuptools
    python312Packages.wheel
    python312Packages.pytest
    python312Packages.black
    python312Packages.mypy
    python312Packages.isort
    python312Packages.flake8

    # ML/AI dependencies
    python312Packages.torch
    python312Packages.transformers

    # Development tools
    git
  ];

  /*
  shellHook = ''
    # Create and activate virtual environment
    if [ ! -d ".venv" ]; then
      python -m venv .venv
    fi
    source .venv/bin/activate

    # Install project and dependencies
    pip install -r requirements.txt

    echo "PersonaFlow development environment activated!"
  '';
  */

  PYTHONPATH = "./";
  PYTHONDONTWRITEBYTECODE = 1;
}
