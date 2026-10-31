.PHONY: app run cli test clean
app:
	./scripts/bundle.sh
run: app
	open "build/MLX Runner.app"
cli:
	swift build -c release --product mlxr
test:
	swift test
clean:
	rm -rf .build build
