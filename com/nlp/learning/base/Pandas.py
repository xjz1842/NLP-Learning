import pandas as pd


def food_map(series):
    if series['food'] == 'A1':
        return 'A'
    elif series['food'] == 'A2':
        return 'A'
    if series['food'] == 'B1':
        return 'B'
    elif series['food'] == 'B2':
        return 'B'
    if series['food'] == 'B3':
        return 'B'
    elif series['food'] == 'C1':
        return 'C'
    elif series['food'] == 'C2':
        return 'C'


if __name__ == "__main__":
    df = pd.read_csv('/Users/zxj/github/machine-learning-stu/titanic.csv')
    # 展示读取数据，默认是前5条
    print(df.head())
    print(df.info())
    print(df.describe())

    print(df.groupby('pclass').sum())
    print(df.groupby('sex')['age'].mean())

    # DataFrame
    data = pd.DataFrame({
        'food': ['A1', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2'],
        'data': [1, 2, 3, 4, 5, 6, 7]
    })

    print(data)

    # apply
    data = pd.DataFrame({'food': ['A', 'A2', 'B1', 'B2', 'B3', 'C1', 'C2'],
                         'data': [1, 2, 3, 4, 5, 6, 7]})
    print(data)

    print(data.apply(food_map, axis='columns'))
