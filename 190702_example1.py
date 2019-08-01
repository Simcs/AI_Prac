print('1.')
for i in range(6):
    print('*', end='')
print()

print('2.')
for i in range(5):
    for j in range(5 - i):
        print('*', end='')
    print()

print('3.')
apart = [ [101, 102, 103, 104], [201, 202, 203, 204], [301, 302, 303, 304], [401, 402, 403, 404] ]
arrears = [101, 203, 301, 404]
for i in range(len(apart)):
    for j in range(len(apart[i])):
        if apart[i][j] not in arrears:
            print('deliver newspaper to room ' + str(apart[i][j]))