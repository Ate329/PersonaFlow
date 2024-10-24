{
  pkgs, ...
}: {
  packages = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.torch
    python312Packages.transformers
    python312Packages.accelerate
    python312Packages.pytest
    python312Packages.pytest-cov
    python312Packages.pytest-xdist
    python312Packages.black
    python312Packages.mypy
    python312Packages.flake8
    python312Packages.bitsandbytes
    pre-commit
    git
  ];

  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
  };

  # Extra shell setup if necessary
  enterShell = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
