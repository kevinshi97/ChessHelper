# Chess Helper

## Packages NEEDED:

The following python packages are needed to run this code:

- pytorch: `pip install pytorch`
- opencv: `pip install opencv-python`
- stockfish: `pip install stockfish`

## Setup for Stockfish:

A very important part of getting the code to work is to set up Stockfish. In addition to running the pip command to install the python Stockfish library, you
need to link the correct Stockfish binary. On lines 65 to 68, there are a few lines to load the correct Stockfish binary. Please use the correct stockfish binary
by uncommenting the correct OS and commenting out the incorrect OS.  

Here are a few command line inputs for you to run to as a demo:

- `python main.py -p assets/test/untitled.png -t b`
- `python main.py -p assets/test/untitled2.png -t w`
- `python main.py -p assets/test/untitled3.png -t w`

The outputs for the demo commands are in the folder sample_outputs.
