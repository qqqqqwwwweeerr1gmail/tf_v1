def is_2d_list(fe):
    return all(isinstance(sublist, list) for sublist in fe) and len(fe) > 0 and all(isinstance(item, (int, float, str)) for sublist in fe for item in sublist)

# Example usage
fe = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
fe = [7, 8, 9]
print(is_2d_list(fe))  # Output: True























