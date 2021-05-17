'''java requires you to specify newline print statements
python does not
to make it print on the same line, specify the end
the same special characters apply in python
print("Hello")
print("Hello", end= " ")
print("Hello" + " world")'''

'''#java requires variable declaration
#python does not
#it also does not differ strings from characters using " and '
x = 2
y = 'world'
z = "Hello"
#python automatically inserts a space wherever there is a comma in a print statement
#concatenation is also different
print("x =", x)
#you can specify what comes after the comma with sep
print("z =", z, sep = " 1 ")'''

'''
#does division like your calculator
#to do integer division use //
a = 3/4
b = 3//4
print("a =", a, end="   ")
print("b = ", b, sep = "")
'''

'''
#casting
c = 3.5
c = int(3.5)
print("c =", c)
#there are built in math concepts
d = min(3, 4, 5, 6, 2)
print("d =", d)
'''

'''
#user input
print("Enter a number: ", end= "")
e = input()
print("You entered:", e)

f = input("Enter a word: ")
print("You said:", f)
#you may have to cast once you have your input
#casting something as a float would be
# a = float(input("enter whatever: "))
'''

'''
#Strings
g = "input"
print("Length of 'input' is:", len(g))
#substrings[start:stop] ... it does not include the stop index
#negative numbers count from the end, where the last letter is -1
print("Substring 1:", g[2:])
print("Substring 2:", g[1:3])
print("Substring 3:", g[-3:-1])
#contains - returns a boolean of if a substring is in the string
print("Contains", ("p" in g))
#strings are immutable in python - every time we create a sting in python, it creates a new object
'''


#lists
#Arraylist - all you need is the two square brackets. then you can add
#objects or remove them
'''list = []
list.append("house")
list.append("mouse")
list.append("blouse")'''
'''
print(list)
print("length:", len(list))
print("Index 1:", list[1])
list.insert(1, "grouse")
print(list)
#you can also specify a range to delete[start:end] with it still excluding the item at end
del(list[1])
print(list)
#we can also access more than one
print("range 1 to 3", list[1:3])
'''

#list operations only in python
'''list2 = []
list2.append("house")
list2.append("mouse")
list2.append("blouse")'''
#we can add list2 to list by extending list, but this does not create a new list ?
'''list.extend(list2)
print(list)'''

'''list3 = list + list2
print(list3)'''
'''list3 = []
list3.append(list)
list3.append(list2)
print(list3)
print(list3[1][2])
#lists are mutable
list3[0] = 4
print(list3)'''

'''
#immutable tuples --> in java, the closest thing is an immutable list
#a list that can not be modified
x = 2, 3, 4, 5
#x[0] = 7 gives an error
print(x)
y = "house", "blouse", "goose"
z = y, 5, "yes"
print(z)
print(z[0])
print(z[1:3])
#to designate a tuple with a single object just follow it with a comma
a = 5,
print(a)
b = 5, 4, 3, 2
print(b[1:3])
print(len(b))
print(3 in b)
print(7 in b)
'''

#review
'''tuples have no bracketing, just commas. they are immutable, cannot be changed
x = 1, 2, 3, 4
lists use square brackets, are mutable and can be changed
y = [1, 2, 3, 4]
strings can use ' or " they are immutable, cannot be changed
z = "house"
all support slicing notation [start:stop]'''

#comparisons for equalities
'''#== checks if two items are equals as far as values go
# "is" checks to see if they are the exact same object/instance
x = input()
y = "house"
print(x == y)
print(x is y)
# < , > , <= , >= , != work the same
#to negate an expression you literally use the word 'not'
#not, and, or are written out
print(not(4 < 5))
print(4 < 5 and 6 < 7)
print( 4 < 5 or 6 > 7)'''

#control statements
'''#since python depends on tabbing and space, this looks different from java
if 4 > 5:
    print('1')
elif 4 > 3:
    print('3')
else:
    print('3')

#loops
x = 0
while x < 5:
    print(x, end = " ")
    x += 1
    #python does not have ++ or --, you must use += or -= and specify the increment

#range automatically takes from 0 up to but not including the limit that you specify
#it increments by 1
for x in range(5):
    print(x, end = " ")
print()
# you can specify where range starts and ends, as well as the increment
#range(start, stop, increment)
for x in range(2, 10, 3):
    print(x, end = " ")

print()
list = [1, 2, 3, 4, 1]
for x in list:
    if x == 3:
        break
        #breaks out of the loop
    print(x, end = " ")

#OR
print()
list = [1, 2, 3, 4, 1]
for x in list:
    if x == 3:
        continue
        #this will hit 3 then continue, printing everything except 3
    print(x, end = " ")
'''

'''#python can combine else statements with repetition statements
for x in range(5):
    if x == 6:
        break
        #x will never be equal to 6
else:
    #if x ends the for loop naturally without hitting a break statment, the else will be excecuted
    print('entered else')
    #if it hits the break statement it will exit the loop'''

#functions
'''
private static int sum(int a, int b, int c) {
    return a + b + c:
}
in java
'''

'''def sum(a, b, c):
    return a + b + c

print(sum(1, 2,  3))
#we can create a variable equal to the sum function and use that variable the same way
#we use sum
mystery = sum
print(mystery(1, 2, 3))
#python supports optional parameters
#in java, you could have two sum functions, one that takes 3 variables and one that takes two
#depending on the number of variables used to call sum(), it would use the corresponding function
#we don't need to do that in python
def add(a, b, c = 0):
    return a + b + c

print(add(3, 4))
print(add(3, 4, 5))'''

#class structure and creating objects
'''
public class Dog {
    private String name;
    private int age;
    
    public Dog(String nm, int a) {
        name = nm;
        age = a;
    }
    public int getAge() {
        return age;
    }
    public String getName() {
        return name;
    }
    public void setAge(int a) {
        age = a;
    }
    public void setName(String nm) {
        name = nm;
    }
    public String toString() {
        return "Dog: \nName: " + name + "Age: " + age;
    }
    
    public static void main(String[] args) {
        Dog d1 = new Dog("Tucker", 1);
        System.out.print(d1);
    }
}
in java
'''

#when we create a class in python it does not have to have the same name as the module name
#python can contain many different classes within the same module
'''class Dog:
    #init is used to create a constructor. self is part of it.
    def __init__(self, name, age):
        #we can create instance variables on the fly
        #the underscore lets other users know that this is a private variable
        #don't change the variable from the outside without using a method
        self._name = name
        self._age = age

    def get_age(self):
        return self._age
    def get_name(self):
        return self._name
    def set_name(self, name):
        self._name = name
    def set_age(self, age):
        self._age = age

    def random():
        return 7

    def __str__(self):
        return "Dog:\nName: " + self._name + "\nAge: " + str(self._age)

#we don't need to use the new operator
d1 = Dog("Tucker", 2)
print(d1)

#when you use self ou designate it as an instance variable or an instance method
#if you omit self then it will be considered a class method
#omiting it is like declaring a method as static in java
#python uses underscores instead of captialization
print(d1.get_age())
print(Dog.random())'''