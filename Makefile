.PHONY: app run cli test release clean
app:
	./scripts/bundle.sh
run: app
	open "build/MLX Runner.app"
cli:
	swift build -c release --product mlxr
test:
	swift test
# Local dry run of the release packaging (unsigned SHA256SUMS): dist/release/
release:
	./scripts/bundle.sh --release
clean:
	rm -rf .build build dist
