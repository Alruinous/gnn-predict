# Usage:
#   make clean-model-files
#   make clean-model-files model_name=<model_name>

.PHONY: clean-model-files
clean-model-files:
	@if [ -n "$(model_name)" ]; then \
		find res/$(model_name)/results -type f -name '*.json' -delete; \
		find res/$(model_name)/logs -type f -delete; \
	else \
		find res -type d -name results -exec find {} -type f -name '*.json' -delete \;; \
		find res -type d -name logs -exec find {} -type f -delete \;; \
	fi
