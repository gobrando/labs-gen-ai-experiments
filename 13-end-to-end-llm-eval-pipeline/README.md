# Experiment 13: End-to-End LLM Evaluation Pipeline

A 6-phase pipeline for evaluating LLM output quality at scale. Given access to a Phoenix observability instance (or trace data), it extracts traces, runs automated quality checks, performs statistical analysis, generates improvement recommendations, and A/B tests prompt changes.

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run all phases on sample data (no API keys needed)
python pipeline.py run --config sample_data/config.yaml
```

This runs phases 1-5 on 20 bundled sample traces and produces:
- `results/traces.json` — Extracted trace data
- `results/eval_results.json` — Automated evaluation results
- `results/sample.json` — Stratified sample selection
- `results/eval_report.md` — Statistical analysis report
- `results/recommendations.md` — Prioritized improvement recommendations

## The 6 Phases

| Phase | Command | What It Does |
|-------|---------|-------------|
| 1. Extract | `pipeline.py extract` | Pull traces from Phoenix API or sample data |
| 2. Evaluate | `pipeline.py evaluate` | Run automated quality dimensions on all traces |
| 3. Sample | `pipeline.py sample` | Stratified sampling for deep review |
| 4. Analyze | `pipeline.py analyze` | Statistical analysis with 95% confidence intervals |
| 5. Improve | `pipeline.py improve` | Map error patterns to prompt improvement strategies |
| 6. Iterate | `pipeline.py iterate` | A/B test prompt changes (requires OpenAI API) |

Run all phases: `python pipeline.py run --config config.yaml`

Run individual phases: `python pipeline.py evaluate --config config.yaml`

## Configuration

Copy `config.example.yaml` and customize:

```yaml
phoenix:
  url: https://your-phoenix-instance.example.com
  project_name: default
  days_back: 60

extraction:
  adapter: generic    # or 'referral', or your custom adapter

evaluation:
  resource_path: resources
  dimensions:
    readability: { enabled: true, max_grade_level: 8.0 }
    url_validity: { enabled: true, skip_validation: false }
    # ... see config.example.yaml for all options

# Optional: A/B testing
iteration:
  versions:
    - name: baseline
      template_path: prompts/v1.txt
    - name: improved
      template_path: prompts/v2.txt
  test_corpus_path: test_corpus.json
  model: gpt-4o
```

## Adapting for Your LLM System

The pipeline uses **adapters** to extract structured data from trace spans. Two built-in adapters are included:

- **`generic`** — Works with any system. Finds the longest output span and tries to parse JSON.
- **`referral`** — Designed for referral/recommendation systems with ReadableLogger + ChatPromptBuilder spans.

To support your system, write a custom adapter. See [`docs/adapters.md`](docs/adapters.md) for details.

## Quality Dimensions

| Dimension | What It Checks | Flags |
|-----------|---------------|-------|
| `output_structure` | Valid JSON with expected keys | `INVALID_JSON`, `ZERO_RESOURCES` |
| `resource_count` | Resource count within bounds | `TOO_FEW_RESOURCES`, `EXCESSIVE_RESOURCES` |
| `url_validity` | HTTP HEAD on resource URLs | `BROKEN_URL`, `HOMEPAGE_ONLY` |
| `readability` | Flesch-Kincaid grade level | `ABOVE_8TH_GRADE` |
| `duplicates` | Fuzzy name/address matching | `DUPLICATE_RESOURCE` |
| `contact_completeness` | Phone/address presence | `MISSING_CONTACT_N` |
| `rag_grounding` | Output vs input context match | `UNGROUNDED_RESOURCE` |
| `location_match` | Geographic consistency | `CROSS_STATE` |

## Methodology

See [`docs/methodology.md`](docs/methodology.md) for the full methodology writeup including statistical approach, sampling strategy, and prompt engineering insights.

## Relationship to Experiment 12

This experiment (13) is the "outer loop" — the full evaluation pipeline. Experiment 12 is the "inner loop" — focused specifically on prompt A/B testing. Phase 6 (Iterate) of this pipeline bundles the same prompt testing modules as Experiment 12.

You can use them independently:
- **Experiment 12 alone:** Quick prompt comparison without the full pipeline
- **Experiment 13 alone:** Full evaluation on production data
- **Both together:** Full pipeline with prompt iteration
