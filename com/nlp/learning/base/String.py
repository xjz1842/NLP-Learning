import keyword

if __name__ == "__main__":
    s = "sad"
    print(str(s))
    print(s.strip())
    print(len(s))
    print(keyword.kwlist)
    print("""
        多行输出
      """)
    print("我是{0},{1}".format("test", "test"))
    print("我是%s,%s" % ("test", "test"))

    choose = input("请输入你的选择(接受 or 拒绝):")

    if choose == '0':
        print("ok")
    else:
        print("false")