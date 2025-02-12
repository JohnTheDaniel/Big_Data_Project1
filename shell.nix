with import <nixpkgs> {};

with pkgs;
let 
  my-python-packages = python-packages: with python-packages; [
    numpy
    scikitlearn
  ];
  python-with-my-packages = python3.withPackages my-python-packages;
in
stdenv.mkDerivation rec {
  name = "Python_Enviroment";
  env = buildEnv {name = name; paths = buildInputs; };
  buildInputs = [python-with-my-packages];
}
