.PHONY: all fmt check test

all: fmt check test

fmt:
	cargo fmt

check:
	cargo clippy --benches -- -D warnings

test:
	cargo test
