def print_hello():
    print('hello python')

def test_lambda(s, t):
    print("input1 == ", s, ", input2 == ", t)

s = 100
t = 200

fx = lambda x, y : test_lambda(x, y)
fy = lambda x, y : print_hello()

fx(300, 400)
fy(1, 2)