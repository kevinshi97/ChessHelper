# Chess Helper

Kevin Shi, Xiqian Liu

## Packages NEEDED

The following python packages are needed to run this code:

- pytorch: `pip install pytorch`
- opencv: `pip install opencv-python`
- stockfish: `pip install stockfish`

## Setup for Stockfish

A very important part of getting the code to work is to set up Stockfish. In addition to running the pip command to install the python Stockfish library, you
need to link the correct Stockfish binary. On lines 65 to 68, there are a few lines to load the correct Stockfish binary. Please use the correct stockfish binary
by uncommenting the correct OS and commenting out the incorrect OS.  

For example, if running on a 64 bit:

```python
stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x64.exe')  # for windows 64 bit
# stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x32.exe')  # for windows 32 bit
# stockfish = Stockfish('assets/stockfish/Mac/stockfish-10-64')           # for mac 64 bit
# stockfish = Stockfish('assets/stockfish/Linux/stockfish_10_x64')        # for linux 64 bit
```

If running on a Mac:

```python
# stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x64.exe')  # for windows 64 bit
# stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x32.exe')  # for windows 32 bit
stockfish = Stockfish('assets/stockfish/Mac/stockfish-10-64')           # for mac 64 bit
# stockfish = Stockfish('assets/stockfish/Linux/stockfish_10_x64')        # for linux 64 bit
```

And if running on a Linux:

```python
# stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x64.exe')  # for windows 64 bit
# stockfish = Stockfish('assets/stockfish/Windows/stockfish_10_x32.exe')  # for windows 32 bit
# stockfish = Stockfish('assets/stockfish/Mac/stockfish-10-64')           # for mac 64 bit
stockfish = Stockfish('assets/stockfish/Linux/stockfish_10_x64')        # for linux 64 bit
```

## Demo

Here are a few command line inputs for you to run to as a demo:

- `python main.py -p assets/test/untitled.png -t b`
- `python main.py -p assets/test/untitled2.png -t w`
- `python main.py -p assets/test/untitled3.png -t w`

The outputs for the demo commands are in the folder sample_outputs.
