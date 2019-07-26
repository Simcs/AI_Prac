for data in range(10):
    print(data, " ", end='')

list_data = [ x**2 for x in range(5) ]
print(list_data)

raw_data = [ [1, 10], [2, 15], [3, 30], [4, 55] ]
all_data = [ x for x in raw_data ]
x_data = [ x[0] for x in raw_data ]
y_data = [ x[1] for x in raw_data ]

print("all_data == ", all_data)
print("x_data == ", x_data)
print("y_data == ", y_data)

even_number = []
for data in range(10):
    if data % 2 == 0:
        even_number.append(data)
even_number = [data for data in range(10) if data % 2 == 0]
print(even_number)

test_data = [ [10, 20, 30], [1, 2, 3], [100, 200, 300] ]
xdata = [ x[1:-1] for x in test_data ]
tdata = [ x[-1] for x in test_data ]
print(xdata)
print(tdata)
