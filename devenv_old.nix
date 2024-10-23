{
  pkgs, ...
}: {
  packages = with pkgs; [
    python312
    python312Packages.pip
    stdenv.cc.cc.lib
    pre-commit
    git
  ];

  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
  };

  enterShell = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
    python -m venv .venv
    source .venv/bin/activate
    pip install torch
    pip install "transformers>=4.45.0"
    pip install accelerate
    pip install pytest pytest-cov pytest-xdist
    pip install black mypy flake8
    pip install bitsandbytes
  '';
}
