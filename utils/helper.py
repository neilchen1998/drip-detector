def print_info(title: str, lines: list):
    """Print the lines in a box made of #'s

    Paramter:
        title: the title
        lines: a list of lines
    """

    # Find the max. length
    max_len = max(len(line) for line in lines)
    border = '#' * (max_len + 4)    # 1 '#' and 1 padding on each side (4 in total)
    div = "# " + '=' * max_len + " #"

    print(border)   # the upper boarder
    print(f"# {title.center(max_len)} #")
    print(div)   # the upper boarder
    for line in lines:
        print(f"# {line.ljust(max_len)} #")
    print(border)   # the lower boarder

    return
