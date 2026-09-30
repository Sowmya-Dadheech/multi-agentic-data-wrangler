.PHONY: install dev test demo bench sandbox-image clean

install:
	pip install -e .

dev:
	pip install -e ".[dev,anthropic,openai]"

test:
	pytest -q

demo:
	wrangle demo

bench:
	python benchmark/run_benchmark.py

sandbox-image:
	docker build -f Dockerfile.sandbox -t wrangler-sandbox:latest .

clean:
	rm -rf .wrangler demo_output benchmark/results/work .pytest_cache
