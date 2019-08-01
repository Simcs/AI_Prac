print('1.')
def add_start_to_end(start, end):
    res = 0
    for i in range(start, end + 1):
        res += i
    return res
print('add_start_to_end(1, 10) == ', add_start_to_end(1, 10))
print('add_start_to_end(10, 9) == ', add_start_to_end(10, 9))

print('2.')
def get_abbr(arr):
    res = []
    for x in arr:
        res.append(x[:3])
    return res
print(get_abbr(['Seoul', 'Daegu', 'Kwangju', 'Jeju']))

print('3.')
def square(i):
    return i ** 2
f = lambda x: square(x)

print('square(2) == ', square(2))
print('f(2) == ', f(2))