from run import solve

def read_test_file(filename: str) -> list[str]:
    with open(filename, 'r') as f:
        return [line.rstrip('\n') for line in f]


def test_depth_2():
    lines = read_test_file('test_data/depth_2.txt')
    result = solve(lines)
    expected = 12521
    
    assert result == expected, f"Test failed: {result} != {expected}"


def test_depth_4():
    lines = read_test_file('test_data/depth_4.txt')
    result = solve(lines)
    expected = 44169

    assert result == expected, f"Test failed: {result} != {expected}"


def main():
    test_depth_2()
    test_depth_4()


if __name__ == "__main__":
    main()
