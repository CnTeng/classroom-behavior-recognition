{
  description = "Classroom Behavior Recognition";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";

    flake-parts.url = "github:hercules-ci/flake-parts";

    devshell = {
      url = "github:numtide/devshell";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    treefmt-nix = {
      url = "github:numtide/treefmt-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs@{ flake-parts, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" ];

      imports = map (n: inputs.${n}.flakeModule) [ "devshell" "treefmt-nix" ];

      perSystem = { pkgs, system, ... }: {
        devshells.default = {
          packages = [
            (pkgs.python3.withPackages (p:
              with p; [
                ipython
                debugpy
                torch
                numpy
                pandas
                opencv4
                matplotlib
                seaborn
                scikit-learn
              ]))
          ];
          env = [{
            name = "TK_LIBRARY";
            value = "${pkgs.tk}/lib/${pkgs.tk.libPrefix}";
          }];
        };

        treefmt = {
          projectRootFile = "flake.nix";
          programs = {
            nixfmt.enable = true;
            prettier.enable = true;
            black.enable = true;
          };
        };
      };
    };
}
