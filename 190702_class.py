def print_name(name):
    print("[def] ", name)

class PrivateMemberTest:
    def __init__(self):
        pass

    def print_name(self, name):
        print("[Test] ", name)

    def call_test(self):
        print_name("KIM")
        self.print_name("KIM")

obj = PrivateMemberTest()
print_name("LEE")
obj.print_name("LEE")
obj.call_test

# obj = PrivateMemberTest("A", "B")
# obj._PrivateMemberTest__name2 = "C"
# print(obj._PrivateMemberTest__name2)