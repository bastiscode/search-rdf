.PHONY: all fmt check test

all: fmt check test

fmt:
	ruff format python
	cargo fmt

check:
	cargo clippy --benches -- -D warnings

test:
	cargo test
