# Filler in LLM answers

Code and results for a paper on filler in LLM answers: text that adds length but no requested information.

- `drybench/`: the package, v1.1.0 (scoring version 9).
- `results/dev_v1.0/`: development runs on a Colab T4; `results/pilot/`: GPT-2 pilot.
- `docs/PROTOCOL.md`: the confirmation run; `docs/ANNOTATION.md`: guide for annotators.

```bash
pip install -e ".[hf,plot,dev]"
python -m pytest -q          # no GPU
```
