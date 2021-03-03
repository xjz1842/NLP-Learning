import numpy as np


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [2, 3, 4]
    print(np.multiply(a, b))
    print(np.average(a))
    print(np.add(a, b))

    # array
    arr_list = [1, 2, 3, 4, 5]
    arr = np.array(arr_list)
    print(arr, type(arr), arr.dtype, arr.ndim, arr.size, arr.shape)

    # slice
    slice_arr = arr[1: 2]
    print(slice_arr)

    # matrix
    matrix = np.array([1, 2, 3, 4, 5])

    print(np.arange(0, 100, 10))

    mask = np.array([0, 1, 1, 0, 1], dtype=bool)
    print(mask)

    random_array = np.random.rand(10)
    print(random_array)

    # arr
    arr = np.array([10, 20, 30, 40, 50])
    print(np.where(arr > 30))

    # mask
    mask = random_array > 0.5
    print(mask)

    # array index
    print(np.where(random_array > 0.5))

    # data type
    arr = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    print(arr, arr.dtype)

    array_copy = arr
    print(array_copy)

    sum1 = np.sum(arr)
    print(sum1)
    print(arr.prod())
    print(arr.var())
    print(arr.round())

    # 最小的缩影位置
    print(arr.argmin())

    # 矩阵乘法
    x = np.array([5, 5])
    y = np.array([2, 2])
    # 矩阵的对应元素相乘
    print(np.multiply(x, y))
    # 矩阵相乘
    print(np.dot(x, y))

    # lineSpace
    arr = np.linspace(0, 10, 10)
    print(arr)

    # sort
    arr = np.array([[1, 0, 6], [1, 7, 0], [2, 3, 1], [2, 4, 0]])
    index = np.lexsort([-1 * arr[:, 0]])
    print(index)
    print(arr[index])

    # shape
    arr = np.arange(10)
    print(arr, arr.shape)
    arr = arr[np.newaxis, :]
    print(arr)
    print(arr.transpose())
    print(arr.squeeze())

    # arr concat
    a = np.array([[1, 2, 3], [4, 5, 6]])
    b = np.array([[7, 8, 9], [10, 11, 12]])
    arr = np.concatenate((b, b))
    print(arr)
    arr = np.hstack((a, b))
    print(arr)
    arr = np.hstack((a, b))
    print(arr)
    arr = arr.flatten()
    print(arr)

    # arr method
    # print(np.zeros((3, 3)))
    # print(np.ones((3, 3) * 8))
    a = np.empty(6, dtype=int)
    a.fill(1)
    print(a)
    likeArr = np.zeros_like(arr)
    print(likeArr)

    print(np.identity(5))

    # random
    print(np.random.rand(3, 2))

    print(np.random.randint(10, size=(5, 4)))

    np.random.seed(100)
    np.set_printoptions(precision=2)

    arr1 = np.arange(10)
    print(np.random.shuffle(arr1))
