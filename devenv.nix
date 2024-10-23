{ pkgs, ... }:

{
  packages = with pkgs; [
    python312
    python312Packages.pip
    git
  ];

  languages.python = {
    enable = true;
    version = "3.11";
    venv.enable = true;
  };

  enterShell = ''
    python -m venv .venv
    source .venv/bin/activate
    pip install torch
    pip install "transformers>=4.45.0"
    pip install accelerate
    pip install pytest pytest-cov pytest-xdist
    pip install black mypy isort flake8
    pip install bitsandbytes
  '';
}
