# Contributing to fraudcost

Thanks for your interest in `fraudcost`! Contributions, bug reports, and ideas are welcome.

## Ways to contribute
- **Report a bug or request a feature** — open an [issue](https://github.com/praveenpolisetty/fraudcost/issues)
  with a clear description and, for bugs, a minimal reproducible example.
- **Improve the docs** — typo fixes and clarifications are always appreciated.
- **Add functionality** — e.g. new calibration methods, AML/graph examples, plotting helpers,
  or a scikit-learn-compatible wrapper (see the roadmap in the README).

## Development setup
```bash
git clone https://github.com/praveenpolisetty/fraudcost.git
cd fraudcost
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install pytest
```

## Running the tests
```bash
pytest -q
```
All tests must pass before a change is merged. CI runs the suite on Python 3.10–3.12 for every push
and pull request.

## Pull request guidelines
1. Fork the repo and create a topic branch (`feature/my-change`).
2. Keep changes focused and add a test for new behavior.
3. Make sure `pytest` passes locally.
4. Open a PR with a clear description of what changed and why.

## Code style
- Keep the library dependency-light (NumPy + scikit-learn only for the core).
- Prefer small, well-documented functions with clear inputs/outputs.
- Match the existing style in `fraudcost/__init__.py`.

## License
By contributing, you agree that your contributions will be licensed under the project's
[MIT License](LICENSE).
