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
    python312Packages.autoflake
    pre-commit
    git
  ];

  languages.python = {
    enable = true;
    version = "3.12";
    venv.enable = true;
  };

  pre-commit.hooks = {
    end-of-file-fixer.enable = true;
    trailing-whitespace.enable = true;
    check-yaml.enable = true;
    check-json.enable = true;
    debug-statements.enable = true;
    commitizen.enable = true;
    black.enable = true;
    flake8 = {
      enable = true;
      args = ["--max-line-length=100"];
    };
    mypy.enable = true;
    autoflake = {
      enable = true;
      args = [
        "--remove-all-unused-imports"
        "--remove-unused-variables"
        "--expand-star-imports"
      ];
    };
  };

  enterShell = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}
