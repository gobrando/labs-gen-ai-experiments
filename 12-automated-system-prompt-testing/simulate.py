#!/usr/bin/env python3
from __future__ import annotations

"""Step 1: Run prompt versions through an LLM on a test corpus.

For each test query, renders each prompt version and calls the OpenAI API.
Saves raw outputs, parsed JSON, and extracted resources.

Usage:
    python simulate.py --config config.yaml [--dry-run] [--limit N]
"""
import json
import time
import logging
import argparse
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from lib.config_loader import load_config
from lib.renderer import render_template
from lib.llm_client import call_openai
from lib.output_parser import parse_json, extract_resources

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_simulation(config: dict, dry_run: bool = False, limit: int = 0) -> list[dict]:
    """Run simulation across all versions and queries."""
    sim_config = config['simulation']
    versions = sim_config['versions']
    model = sim_config.get('model', 'gpt-4o')
    temperature = sim_config.get('temperature', 0.7)
    template_format = sim_config.get('template_format', 'jinja2')
    message_format = sim_config.get('message_format', 'system_only')
    rate_limit_delay = sim_config.get('rate_limit_delay', 2.0)
    resource_path = config.get('evaluation', {}).get('resource_path', 'resources')

    # Load templates
    templates = {}
    for v in versions:
        name = v['name']
        path = Path(v['template_path'])
        if not path.exists():
            raise FileNotFoundError(f"Template not found for {name}: {path}")
        templates[name] = path.read_text()
        logger.info(f"Loaded template '{name}': {len(templates[name])} chars")

    # Load test corpus
    corpus_path = Path(sim_config['test_corpus_path'])
    if not corpus_path.exists():
        raise FileNotFoundError(f"Test corpus not found: {corpus_path}")
    with open(corpus_path) as f:
        test_corpus = json.load(f)

    if limit > 0:
        test_corpus = test_corpus[:limit]

    version_names = [v['name'] for v in versions]
    logger.info(f"Processing {len(test_corpus)} queries x {len(version_names)} versions")

    results = []

    for i, query in enumerate(test_corpus):
        query_id = query.get('id', query.get('trace_id', f'query_{i+1:03d}'))
        user_query = query.get('query', query.get('user_query', ''))
        resources_context = query.get('resources_context', '')
        location = query.get('location', '')

        logger.info(f"[{i+1}/{len(test_corpus)}] id={query_id} query={user_query[:60]}...")

        entry = {
            'query_id': query_id,
            'user_query': user_query,
            'location': location,
            'resources_context_length': len(resources_context),
        }

        # Build template variables from query data
        variables = {
            'query': user_query,
            'user_query': user_query,
            'location': location,
            'resources_context': resources_context,
            # Common Jinja2 variable names
            'supports': resources_context.split('\n\n') if resources_context else [],
            'response_json': query.get('response_json', '{}'),
        }
        # Include any extra fields from query
        for key, val in query.items():
            if key not in variables:
                variables[key] = val

        for vname in version_names:
            rendered = render_template(templates[vname], variables, template_format)

            if dry_run:
                entry[vname] = {
                    'rendered_prompt_length': len(rendered),
                    'rendered_prompt_preview': rendered[:300],
                }
                continue

            try:
                logger.info(f"  Calling {vname}...")
                resp = call_openai(
                    system_prompt=rendered,
                    model=model,
                    temperature=temperature,
                    message_format=message_format,
                )
                parsed, parse_err = parse_json(resp['content'])
                resources = extract_resources(parsed, resource_path)

                entry[vname] = {
                    'raw_content': resp['content'],
                    'parsed_json': parsed,
                    'parse_error': parse_err,
                    'resources': resources,
                    'resource_count': len(resources),
                    'usage': resp['usage'],
                    'finish_reason': resp['finish_reason'],
                }
            except Exception as e:
                logger.error(f"  {vname} API error: {e}")
                entry[vname] = {'error': str(e)}

            time.sleep(1)

        results.append(entry)

        if i < len(test_corpus) - 1:
            time.sleep(rate_limit_delay)

    return results


def main():
    parser = argparse.ArgumentParser(description='Run prompt simulation')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--dry-run', action='store_true', help='Render prompts without API calls')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of queries (0=all)')
    args = parser.parse_args()

    config = load_config(args.config)
    results = run_simulation(config, dry_run=args.dry_run, limit=args.limit)

    # Save results
    output_dir = Path(config['simulation']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'simulation_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved {len(results)} results to {output_path}")

    # Summary
    if not args.dry_run:
        version_names = [v['name'] for v in config['simulation']['versions']]
        for vname in version_names:
            valid = sum(1 for r in results if r.get(vname, {}).get('parsed_json') is not None)
            avg = sum(r.get(vname, {}).get('resource_count', 0) for r in results) / max(len(results), 1)
            logger.info(f"{vname}: valid_json={valid}/{len(results)}, avg_resources={avg:.1f}")


if __name__ == '__main__':
    main()
