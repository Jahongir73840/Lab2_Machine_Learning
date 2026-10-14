# Lab 2: A Neural Network from Scratch

Solution for the lab assignment "A neural network from scratch".
The notebook follows the lab manual and uses simple NumPy functions.

## Files to open

- **Lab2_Neural_Network.ipynb** - the main file with explanations, formulas,
  code, saved results, and plots. Open it in Jupyter, VS Code, or Colab.
- **Lab2_Neural_Network.html** - a browser version that does not require running Python.
- **lab2.py** - the same code and explanations in a regular Python file.
- **results/** - loss curves, gradient checks, accuracy results, and timings.

The notebook has already been executed. It covers all seven parts of the
assignment, including matrix shapes and gradient derivation using the chain
rule. The manual asks for the network design and derivatives on paper;
sections 2 and 4 of the notebook can be used as a reference.

## Results from this run

- Dataset split: 1,257 training and 540 test examples, stratified, with seed 42.
- Gradient check: 9/10 passed for W=16 and 10/10 for W=64. At least 9/10 are required.
- All 6 combinations were trained for 50 epochs with a mini-batch size of 32.
- Best observed test accuracy: **90.74%**, with W=64 and learning rate=0.1.
- With learning rates of 0.001 and 0.01, accuracy stayed around 10%: the model
  made little progress in 50 epochs. These results are saved and discussed
  in the conclusion.

The manual's N(0, 0.01) notation is interpreted as a mean of 0 and a standard
deviation of 0.01: `rng.normal(0, 0.01, ...)`. This is stated in the notebook.

## How to run

Install the dependencies in your Python environment, then run the script:

```powershell
python -m pip install -r requirements.txt
python lab2.py
```

A local environment has already been prepared on the original computer.
On that computer, the script can also be run with:

```powershell
.\.venv\Scripts\python.exe lab2.py
```

For the notebook, select the environment as the Python kernel and choose
**Run All**. In Colab, upload the `.ipynb` file and run all cells.
Running the code again overwrites the files in `results/`.
Timing measurements may differ slightly from the saved values.

## Result files

- `loss_curves.png` - training loss curves for all 6 combinations.
- `gradient_check.csv` - numerical and analytical gradients, with relative errors.
- `metrics.csv` - final loss and accuracy, plus total training time including logging.
- `epoch_log.csv` - training loss and accuracy after each epoch.
- `timing.csv` - median forward and backward pass times over 100 measurements.
- `timing_samples.csv` - the individual timing measurements.

CSV files can be opened as plain text or imported into Excel.
