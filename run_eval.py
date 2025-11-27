import litellm
from lighteval.__main__ import app

# Monkeypatch litellm.completion to handle custom model names with reasoning effort suffixes
# e.g. "model-high" -> model="model", reasoning_effort="high"
original_completion = litellm.completion

def patched_completion(*args, **kwargs):
    # Check if model name contains reasoning effort suffix
    if "model" in kwargs:
        model_name = kwargs["model"]
        # Defined reasoning efforts supported by OpenRouter/LiteLLM
        efforts = ["high", "medium", "low", "minimal", "none"]

        for effort in efforts:
            suffix = f"-{effort}"
            if model_name.endswith(suffix):
                # Strip the suffix to get the real model name
                real_model_name = model_name[:-len(suffix)]

                # Update kwargs
                kwargs["model"] = real_model_name
                if "reasoning_effort" not in kwargs:
                    kwargs["reasoning_effort"] = effort

                break # Stop checking other efforts

    return original_completion(*args, **kwargs)

litellm.completion = patched_completion

if __name__ == "__main__":
    app()

