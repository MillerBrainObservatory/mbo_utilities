def _print_params(params, indent=5):
    for k, v in params.items():
        # if value is a dictionary, recursively call the function
        if isinstance(v, dict):
            print(" " * indent + f"{k}:")
            _print_params(v, indent + 4)
        else:
            print(" " * indent + f"{k}: {v}")
