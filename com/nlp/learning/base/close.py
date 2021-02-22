def func():
    print("func is run")

func()

ret = func()

def out_func():
    data = 520

    def inner_func():
        return data
    return inner_func

if __name__ == "__main__":

    print(id(func()),id(ret))

    print(out_func()())
