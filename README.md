This repository implements the eSCP algorithm given in [^1] for the purpose of running several experiments detailed in a paper[^2] that studies the continuity of this algorithm. The three modules 'src/example_{one, two, three}.py' reproduce the plots from [^2].

[^1]: Shi, H., Yang, L., Chi, J., Butler, T., Wang, H., Bingham, D., & Estep, D. (2025). Nonparametric Bayesian Calibration of Computer Models. *arXiv preprint arXiv:2509.22597.*

[^2]: Prasadan A., Bingham, D., & Estep, D. (2026). Continuity of the Solution of a Non-Parametric Bayesian Statistical Calibration Procedure. *arXiv preprint arXiv:2603.20665.*

## Getting Started
 
1. Install **uv**, a package manager. [See instructions here](https://docs.astral.sh/uv/getting-started/installation/).

2. Clone the repository and navigate to it.

        git clone https://github.com/akprasadan/escp-continuity.git

        cd escp-continuity

3. Install the project and its dependencies.

        uv sync

## Usage

Now you can run modules with **uv** in a virtual environment, e.g., 

        uv run python -m src.example_one

        uv run python -m tests.check_pushforward

You can also run all unit tests.

        uv run pytest

## Directory Structure

- The core functions and examples are stored in `src/`.
- A dataset needed for `src/example_three.py` is given in `data/Concrete_Data.xls`, and running this example will also store a processed version of the dataset as `data/processed_concrete_data.csv`. The original dataset is due to [^3] and can be downloaded from the [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength).
- Results from running the examples are stored in `plots/`.
- The code is extensively tested in `tests/`. Three tests, `tests/check_pushforward.py`, `tests/check_concrete_pushforward.py`, `tests/test_probs_to_mesh.py` include visual tests and must be manually run. Their results are stored in `plots/tests/`.

[^3]: Yeh, I. C. (1998). Modeling of strength of high-performance concrete using artificial neural networks. *Cement and Concrete research, 28*(12), 1797-1808.


## License

[MIT](https://choosealicense.com/licenses/mit/)


