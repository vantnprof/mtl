def find_config(target_params: int, D1: int, D2: int, D3: int, d3: int):
    """
    Finds integer values d1, d2, r1, r2, r3 such that:
    d1 + r1*d1 + r1*D1 + r1 + d2 + r2*d2 + r2*D2 + r2 + d3 + r3*d3 + r3*D3 + r3 = k**2 * D3 * d3
    with constraints d1 = d2 and r1 = r2.
    Returns (d1, d2, r1, r2, r3) or, if not found, the configuration with the nearest sum.
    """
    max_val = 200  # limit search space for practical reasons
    best = None
    best_diff = float('inf')
    for d1 in range(1, max_val):
        d2 = d1
        for r1 in range(0, max_val):
            r2 = r1
            for r3 in range(0, max_val):
                lhs = (
                    d1 + r1*d1 + r1*D1 + r1 +
                    d2 + r2*d2 + r2*D2 + r2 +
                    d3 + r3*d3 + r3*D3 + r3
                )
                diff = abs(lhs - target_params)
                if diff < best_diff:
                    best = (d1, d2, r1, r2, r3, lhs)
                    best_diff = diff
                if lhs == target_params:
                    return d1, d2, r1, r2, r3, lhs
    # If not found, return nearest configuration (with sum as 6th value)
    if best is not None:
        return best
    raise ValueError("No solution found for the given parameters.")


if __name__ == "__main__":
    # conv2d-like test cases: (kernel, H, W, Cin, Cout)
    test_cases = [
        (3, 32, 32, 3, 16),
        (5, 28, 28, 1, 32),
        (1, 64, 64, 16, 16),
        (3, 7, 7, 64, 128),
        (3, 224, 224, 3, 64),
        (5, 14, 14, 128, 256),
        (1, 56, 56, 64, 64),
        (3, 16, 16, 32, 64),
    ]
    results = []
    for idx, (k, H, W, Cin, Cout) in enumerate(test_cases):
        try:
            config = find_config(k, H, W, Cin, Cout)
            found = (len(config) == 6 and config[5] == k**2 * Cin * Cout)
        except ValueError:
            config = ("-", "-", "-", "-", "-", "-")
            found = False
        results.append((k, H, W, Cin, Cout, *config, "OK" if found else "Nearest"))

    # Print as table
    header = [
        "k", "H", "W", "Cin", "Cout", "d1", "d2", "r1", "r2", "r3", "Sum", "Result", "Target"
    ]
    print("{:<3} {:<4} {:<4} {:<5} {:<5} {:<4} {:<4} {:<4} {:<4} {:<4} {:<8} {:<12} {:<8}".format(*header))
    print("-" * 85)
    for row in results:
        k, H, W, Cin, Cout, d1, d2, r1, r2, r3, sum_found, result = row
        target_sum = int(k)**2 * int(Cin) * int(Cout)
        print("{:<3} {:<4} {:<4} {:<5} {:<5} {:<4} {:<4} {:<4} {:<4} {:<4} {:<8} {:<12} {:<8}".format(
            k, H, W, Cin, Cout, d1, d2, r1, r2, r3, sum_found, result, target_sum
        ))

    # Show number of unique targets and found configurations
    targets = set()
    found_configs = set()
    for (k, H, W, Cin, Cout, d1, d2, r1, r2, r3, sum_found, result) in results:
        targets.add((k, H, W, Cin, Cout))
        if result == "OK":
            found_configs.add((d1, d2, r1, r2, r3, H, W, Cin))
    print("\nNumber of unique targets:", len(targets))
    print("Number of unique found configurations:", len(found_configs))