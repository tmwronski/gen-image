# Genetic Image *(gen-image)*

Image reproduction using a genetic algorithm.

<p align="center">
    <img src="assets/target.jpg">
    <img src="assets/result.gif">
    <img src="assets/result.jpg">
</p>

## Instalation

For Ubuntu 18.04

```
$ python3 -m venv venv
$ source venv/bin/activate
$ install -r requirements.txt
```

## Usage

```
$ python3 gen-image [d|v|q] [path]

-> --dump | -d      enable dumping each 1000 genertions
-> --quiet | -q     disable verbosity
-> --verbose | -v   enable verbosity (default)
```