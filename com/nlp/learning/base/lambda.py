from functools import reduce

if __name__ == "__main__":
    g = lambda x: x * 2
    print(g(2))
    g = (lambda x: x * 2)(4)
    print(g)

    for n in ["qi", "yue", "july"]:
        print(len(n))

    name_len = map(len, ["qi", "yue", "july"])
    print(list(name_len))

    number_list = range(-5, 5)
    less_than_zero = list(filter(lambda x: x < 0, number_list))
    print(less_than_zero)

    number_list = range(-5, 5)
    greater_than_zero = list(filter(lambda x: x > 0, number_list))
    average = reduce(lambda x, y: x + y, greater_than_zero) / len(greater_than_zero)
    print(average)
