#!/usr/bin/env python
# coding: utf-8

# In[2]:


print("helo")


# In[3]:


a=int(input("input first number"))
b=int(input("input first number"))
sum=a+b
print(sum)


# In[59]:


def sum(a,b):
 sm=a+b
 return sm
sum(5,8)


# In[58]:


class car:
    def __init__(self,model,company,color):
        self.model = model
        self.company = company
        self.color = color
    def show(self):
        print("Comapany name: ", self.company)
        print("Model no: ", self.model)
        print("car color: ", self.color)

car1 = car(122,"honda","lack")
car1.show()


# In[5]:


dic = {"name":"sunail","roll no":122,"marks":[23,4,57,7]}
dic["name"]
for i,j in dic.items():
    print(i, j)


# In[6]:


dic = {"name":"sunail","roll no":122,"marks":[23,4,57,7]}
for i in dic.items():
    print(i)


# In[8]:


dic = {"name":"sunail","roll no":122,"marks":[23,4,57,7]}
for i in dic.keys():
    print(i)


# In[9]:


dic = {"name":"sunail","roll no":122,"marks":[23,4,57,7]}
for i in dic.values():
    print(i)


# In[10]:


pip install numpy


# In[11]:


import numpy


# In[12]:


numpy.array([389,3823])


# In[13]:


import numpy as np


# In[14]:


np.array([389,3879])


# In[20]:


data=[0,1,2,3,4,5]
a=np.array(data,"float64")
a


# In[22]:


data=[0,1,2.0,3,4,5]
a=np.array(data,)
a


# In[24]:


data=[0,1,2,3,4,5]
a=np.array(data)
a


# In[25]:


a.dtype


# In[27]:


a.nd


# In[29]:


a.nd in


# In[30]:


a.shape


# In[ ]:


b=array([[2,3],[4,5]])


# In[31]:


b.shape


# In[32]:


a.size


# In[33]:


a.itemsize


# In[1]:


np.shape


# In[2]:


a.shape


# In[3]:


pip install numpy


# In[4]:


import numpy


# In[5]:


import numpy as np


# In[6]:


data=[0,1,2,3,4,5]
a=np.array(data,"float64")
a


# In[7]:


a.size


# In[8]:


a.shape


# In[9]:


a.sizeitems


# In[10]:


a.sizeitems


# In[11]:


a.itemsize


# In[12]:


a


# In[13]:


a[0]


# In[14]:


a[4]


# In[15]:


a[4:5]


# In[16]:


a[1:5]


# In[3]:


a[1:8]


# In[18]:


a.append([9])


# In[19]:


a.apend([9])


# In[4]:


a.append([9])


# In[5]:


a[2:7]


# In[6]:


a.dtype


# In[7]:


data=[0,1,2,3,4,5]
a=np.array(data,"float64")
a


# In[8]:


pip install numpy


# In[9]:


import numpy as np


# In[10]:


data=[0,1,2,3,4,5]
a=np.array(data,"float64")
a


# In[11]:


a.shape


# In[12]:


a.size


# In[13]:


a.itemsize


# In[14]:


a[1:5]


# In[15]:


a.dtype


# In[16]:


a.nd


# In[17]:


a.d


# In[18]:


data=[[1,2,3],[4,5,6],[7,8,9]]
b=np.array(data,"float32")
b


# In[19]:


b[0]


# In[20]:


b.dtype


# In[21]:


b[2]


# In[22]:


b[2.3]


# In[23]:


b[2,3]


# In[24]:


b[0,1]


# In[25]:


b


# In[26]:


b[1:3,1:3]


# In[27]:


b.itemsize


# In[28]:


b.size


# In[29]:


b.shape


# In[30]:


b.mini()


# In[32]:


b.reveal


# In[33]:


np.argmax(b)


# In[35]:


np.argmin(b)


# In[36]:


np.sort(b)


# In[37]:


np.append(b,[11,12,13])


# In[38]:


np.arrange(1,8,3)


# In[39]:


np.arrang(1,8,3)


# In[40]:


np.arrange(0,2,2)


# In[41]:


np.arrange(1:8:3)


# In[42]:


np.zero(4,4)


# In[43]:


np.zero((4,4))


# In[44]:


np.zero(4,4)


# In[ ]:





# In[45]:


np.ones((3,5))


# In[46]:


np.zeros((4,4))


# In[47]:


c=np.full((2,2),"a")


# In[48]:


c


# In[49]:


d=eye(3)


# In[50]:


d=eyes(3)


# In[52]:


np.zeros(4,4)


# In[53]:


np.zeros(4,4)


# In[54]:


np.zeros((4,4))


# In[55]:


np.eye((3))


# In[56]:


d=eye((3))


# In[57]:


d=n.eye((3))


# In[58]:


d=np.eye((3))


# In[59]:


d


# In[60]:


e=np.random.random(2,2)


# In[61]:


e=np.random.random((2,2))


# In[62]:


e


# In[63]:


print(np.cos(e))


# In[66]:


f=np.ramdon.random((4,4))


# In[68]:


f=np.ramdom.random((4,4))


# In[69]:


f=np.random.random((4,4))


# In[70]:


f


# In[71]:


print(np.sum(f,axis=0))


# In[72]:


print(np.sum(f,axis=1))


# In[73]:


np.arrange((1,8,3))


# In[74]:


np.arange((1,8,3))


# In[75]:


np.arrange((1:8:3))


# In[76]:


np.arrange(1,8,3)


# In[77]:


np.arrang(1,8,3)


# In[78]:


np.arrange((1,8,3))


# In[79]:


np.arrange(1,8,3)


# In[1]:


np.arrange(1,7)


# In[2]:


data=[1,2,3,4,5,6]
data


# In[3]:


pip install pandas


# In[4]:


import pandas as pd


# In[10]:


s=pd.series([1, 3, 5, 6, 6, 8]) 
print(s)


# In[8]:


s=pd.series([1, 3, 5, 6, 6, 8])  
print(s)


# In[9]:


s=pd.series([1,23,6,7,9])


# In[12]:


s=pd.series[[1, 3, 5, 6, 6, 8]]
print(s)


# In[ ]:





# In[2]:


pip install pandas


# In[3]:


import pandas as pd


# In[4]:


data = {
    'apples': [3, 2, 0, 1], 
    'oranges': [0, 3, 7, 2]
}


# In[5]:


purchases = pd.DataFrame(data)

purchases


# In[6]:


purchases = pd.DataFrame(data, index=['June', 'Robert', 'Lily', 'David'])

purchases


# In[7]:


purchases.loc['June'] #return the values at the certain location


# In[22]:


# Read data from CSV file

csv_path = 'C:/Users/abdul/Downloads/GB-Data/GB-Data/Labs/titanic_data.csv' #can use multiple formats e.g. xlsx,json etc
df = pd.read_csv(csv_path)
df


# In[21]:


# Input a five-digit number lab task3
number = int(input("Enter a five-digit number: "))

# Generate the reverse
reverse = 0
temp = number

while temp > 0:
    digit = temp % 10
    reverse = (reverse * 10) + digit
    temp = temp // 10

# Display the reverse
print("The reverse of the number is:", reverse)


# In[1]:


student=["ali","raza","kazim"]


# In[ ]:


roll_no=[2009,2010,2011]


# In[ ]:


student_roll_no=student + roll_no


# In[ ]:


print(Student_roll_no)


# In[1]:


student=["ali","raza","kazim"]
roll_no=[2009,2010,2011]
student_roll_no=student + roll_no
print(student_roll_no)


# In[2]:


student_roll_no


# In[9]:


nume1=float(input("input first number:"))
nume2=float(input("input second number:"))
sum=nume1+nume2
print(sum)


# num1=int(input("input 1st number:"))

# In[13]:


#lab task 3 basic calculations
num1=int(input("input 1st number:"))
num2=int(input("input 2nd number:"))
coution=num1//num2
modulus=num1%num2
division=num1/num2
sum=num1+num2
sub=num1-num2
print("caution:",coution)
print("remaider:",modulus)
print("division:",division)
print("sum of two numbers:",sum)
print("substraction of two numbers:",sub)


# In[19]:


#lab task 3 circle calculations
radius= int(input("input the radius:"))
diameter=radius*2
pi=22/7
circumference=2*pi*radius
area=pi*radius**2
print("diameter of circle:",diameter)
print("circumference of circle:",circumference)
print("area of circle:", area)


# In[34]:


#lab task 3 square
print(" "*2,"*"*9)
for i in range(6):
 print(" "*2, "*", " "*5,"*")
print(" "*2,"*"*9)


# In[56]:


#lab task 3 square
print(" "*9,"*"*3)
for i in range(6):
 print(" "*(8-i*2-1),"*", " "*(2+i*2-1),"*")
print(" "*8,"*"*3)


# In[57]:


def print_box():
    for i in range(5):
        if i == 0 or i == 4:
            print("*********")
        else:
            print("*       *")

def print_oval():
    for i in range(6):
        if i == 0 or i == 5:
            print("  *******")
        elif i == 1 or i == 4:
            print(" *       *")
        else:
            print("*         *")

def print_arrow():
    for i in range(6):
        if i == 0:
            print("    *")
        elif i == 1:
            print("   ***")
        elif i == 2:
            print("  *****")
        else:
            print("    *")

def print_diamond():
    for i in range(7):
        if i < 4:
            print(" " * (3 - i) + "*" * (2 * i + 1))
        else:
            print(" " * (i - 3) + "*" * (13 - 2 * i))

print("Box:")
print_box()

print("\nOval:")
print_oval()

print("\nArrow:")
print_arrow()

print("\nDiamond:")
print_diamond()


# In[60]:


for i in range(8, 90, 3):
    print(i)


# In[61]:


#lab task 12 1 append and add list
appointments = ['9:00', '10:30', '14:00', '15:00', '15:30']

# Using append() method
appointments.append('16:30')

# Using + operator
new_appointments = appointments + ['16:30']

print("Modified list using append():", appointments)
print("New list using + operator:", new_appointments)



# In[80]:


#lab task 12 2 square
ids=[4353, 2314, 2956, 3382, 9362, 3900]
ids.remove(3382)
print(ids)
print(ids.index(9362))
ids.insert(4,4499)
print(ids)
ids=ids+[5566,1830]
print(ids)
ids.reverse()
print(ids)
ids.sort()
print(ids)


# In[81]:


# lab task 12 3

numbers = []

# Inputting numbers into the list
for i in range(10):
    number = float(input("Enter number {}: ".format(i+1)))
    numbers.append(number)

# Finding the largest number and its location
largest_number = numbers[0]
largest_index = 0

for i in range(1, len(numbers)):
    if numbers[i] > largest_number:
        largest_number = numbers[i]
        largest_index = i

# Printing the largest number and its location
print("Largest number:", largest_number)
print("Location within the list:", largest_index)


# In[82]:


# lab task 12
numbers = []

# Reading and checking for duplicates
for _ in range(20):
    number = int(input("Enter a number between 10 and 100 (inclusive): "))

    if number >= 10 and number <= 100:
        if number not in numbers:
            numbers.append(number)
            print(number)
        else:
            print("Duplicate number:", number)
    else:
        print("Invalid number!")


#    # Initialize a 3x3 matrix
# matrix = []
# 
# # Read the matrix from the user
# print("Enter the elements of the matrix:")
# 
# for _ in range(3):
#     row = []
#     for _ in range(3):
#         element = float(input())
#         row.append(element)
#     matrix.append(row)
# 
# # Print the original matrix
# print("Original Matrix:")
# for row in matrix:
#     for element in row:
#         print(element, end=" ")
#     print()
# 
# # Calculate the transpose of the matrix
# transpose = [[matrix[j][i] for j in range(3)] for i in range(3)]
# 
# # Print the transpose matrix
# print("Transpose Matrix:")
# for row in transpose:
#     for element in row:
#         print(element, end=" ")
#     print()
# 

# In[85]:


# Initialize a 3x3 matrix¶
matrix = []

# Read the matrix from the user
print("Enter the elements of the matrix:")

for _ in range(3): 
    row = [] 
    for _ in range(3): 
        element = float(input()) 
        row.append(element) 
        matrix.append(row)

# Print the original matrix
print("Original Matrix:") for row in matrix: for element in row: print(element, end=" ") print()

Calculate the transpose of the matrix
transpose = [[matrix[j][i] for j in range(3)] for i in range(3)]

# Print the transpose matrix
print("Transpose Matrix:") for row in transpose: for element in row: print(element, end=" ") print()


# In[1]:


# Define a function that converts Fahrenheit to Celsius lab task
def to_celsius(x):
    return (x-32) * 5/9

# Create a temperature conversion table using string formatting
for x in range(0, 101, 10):
    print("{:>3} F | {:>6.2f} C".format(x, to_celsius(x)))


# In[8]:


matrix = []
print("Enter the elements of the matrix row by row:")
for _ in range(3):
    row = list(map(int, input().split()))
    matrix.append(row)
return matrix

# Function to calculate the transpose of a matrix
def transpose_matrix(matrix):
    transpose = [[0 for _ in range(3)] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            transpose[j][i] = matrix[i][j]
    return transpose

# Read the matrix from the user
print("Enter a 3x3 matrix:")
matrix = read_matrix()

# Calculate the transpose of the matrix
transpose = transpose_matrix(matrix)

# Print the original matrix
print("Original Matrix:")
for row in matrix:
    print(*row)

# Print the transpose matrix
print("Transpose Matrix:")
for row in transpose:
    print(*row)


# # Function to read a matrix from the user
# def read_matrix(rows, cols):
#     matrix = []
#     print(f"Enter {rows} rows and {cols} columns:")
#     for i in range(rows):
#         row = []
#         for j in range(cols):
#             value = int(input(f"Enter element at position ({i+1}, {j+1}): "))
#             row.append(value)
#         matrix.append(row)
#     return matrix
# 
# # Function to calculate the transpose of a matrix
# def transpose_matrix(matrix):
#     transpose = []
#     for j in range(len(matrix[0])):
#         row = []
#         for i in range(len(matrix)):
#             row.append(matrix[i][j])
#         transpose.append(row)
#     return transpose
# 
# # Read a 3x3 matrix from the user
# matrix = read_matrix(3, 3)
# 
# # Calculate the transpose of the matrix
# transpose = transpose_matrix(matrix)
# 
# # Print the original matrix
# print("Original Matrix:")
# for row in matrix:
#     print(row)
# 
# # Print the transpose matrix
# print("Transpose Matrix:")
# for row in transpose:
#     print(row)

# In[9]:


# Function to read a matrix from the user
def read_matrix(rows, cols):
    matrix = []
    print(f"Enter {rows} rows and {cols} columns:")
    for i in range(rows):
        row = []
        for j in range(cols):
            value = int(input(f"Enter element at position ({i+1}, {j+1}): "))
            row.append(value)
        matrix.append(row)
    return matrix

# Function to calculate the transpose of a matrix
def transpose_matrix(matrix):
    transpose = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        transpose.append(row)
    return transpose

# Read a 3x3 matrix from the user
matrix = read_matrix(3, 3)

# Calculate the transpose of the matrix
transpose = transpose_matrix(matrix)

# Print the original matrix
print("Original Matrix:")
for row in matrix:
    print(row)

# Print the transpose matrix
print("Transpose Matrix:")
for row in transpose:
    print(row)


# In[13]:


# lab 12
def read_matrix():
    matrix = []
    print("Enter 3x3 matrix (row-wise):")
    for _ in range(3):
        row = list(map(int, input().split()))
        if len(row) != 3:
            raise ValueError("Invalid matrix. Please enter 3 elements per row.")
        matrix.append(row)
    return matrix

def multiply_matrices(matrix1, matrix2):
    result = [[0 for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def print_matrix(matrix, title):
    print(title)
    for row in matrix:
        print(" ".join(map(str, row)))
    print()

def main():
    print("Enter the first matrix:")
    matrix1 = read_matrix()

    print("Enter the second matrix:")
    matrix2 = read_matrix()

    try:
        result_matrix = multiply_matrices(matrix1, matrix2)
        print_matrix(matrix1, "Matrix 1:")
        print_matrix(matrix2, "Matrix 2:")
        print_matrix(result_matrix, "Product Matrix:")
    except ValueError as e:
        print("Error:", e)

if __name__ == "__main__":
    main()


# 
# 
# 
# # Read the size of the matrix from the user
# rows = int(input("Enter the number of rows in the matrix: "))
# cols = int(input("Enter the number of columns in the matrix: "))
# 
# # Create an empty matrix
# matrix = []
# for _ in range(rows):
#     row = []
#     for _ in range(cols):
#         # Populate the matrix with data from the user
#         element = int(input("Enter an element: "))
#         row.append(element)
#     matrix.append(row)
# 
# # Print the original matrix
# print("Original matrix:")
# for row in matrix:
#     for element in row:
#         print(element, end=" ")
#     print()
# 
# # Calculate the transpose of the matrix
# transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
# 
# # Print the transpose matrix
# print("Transpose matrix:")
# for row in transpose:
#     for element in row:
#         print(element, end=" ")
#     print()
# 

# In[14]:


#lab 12

# Read the size of the matrix from the user
rows = int(input("Enter the number of rows in the matrix: "))
cols = int(input("Enter the number of columns in the matrix: "))

# Create an empty matrix
matrix = []
for _ in range(rows):
    row = []
    for _ in range(cols):
        # Populate the matrix with data from the user
        element = int(input("Enter an element: "))
        row.append(element)
    matrix.append(row)

# Print the original matrix
print("Original matrix:")
for row in matrix:
    for element in row:
        print(element, end=" ")
    print()

# Calculate the transpose of the matrix
transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]

# Print the transpose matrix
print("Transpose matrix:")
for row in transpose:
    for element in row:
        print(element, end=" ")
    print()


# In[15]:


#lab 11
import random
from collections import Counter

def mean(lst):
    return int(sum(lst) / len(lst))

def median(lst):
    sorted_lst = sorted(lst)
    mid = len(lst) // 2
    if len(lst) % 2 == 0:
        return int((sorted_lst[mid - 1] + sorted_lst[mid]) / 2)
    else:
        return int(sorted_lst[mid])

def mode(lst):
    count_dict = Counter(lst)
    max_count = max(count_dict.values())
    modes = [num for num, count in count_dict.items() if count == max_count]
    return modes[0]

def histogram(lst):
    freq_dict = Counter(lst)
    print("Frequency | Histogram")
    for num, freq in sorted(freq_dict.items()):
        print(f"{num:9d} | {'*' * freq}")

# Generate random numbers
random_numbers = [random.randint(1, 100) for _ in range(99)]

# Compute statistics
mean_value = mean(random_numbers)
median_value = median(random_numbers)
mode_value = mode(random_numbers)

# Print results
print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)
print("Histogram:")
histogram(random_numbers)


# In[1]:


img_x,img_y=28,28
from 


# In[5]:


import pandas as pd
x_train.shape


# In[6]:


x_train=x_train.reshape(x_train.shape(0),28,28,1)
x_test=x_test.reshape(x_test.shape(0),28,28,1)


# In[7]:


x_train=x_train/255.0


# In[17]:


get_ipython().system('pip install keras')


# In[20]:


from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MinPooling2D
from keras.models import Sequential
model=sequential()
model.add(Cov2D(32, kernel_size=(5,5),
               strides=(1,1),
               activation='relu',
               input_shape=(28,28,1)))
model.add(MinPoolin2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='catagorical_crossentropy',optimizer='adam',metrics=['accuracy'])
history=model.fit(x_train,y_train,batch_size=128,epochs=10, verbose=1)


# In[1]:


import pandas as pd

data = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]

df = pd.DataFrame(data, columns=['Name', 'Age'])

print(df)


# In[2]:


import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35]}

df = pd.DataFrame(data)

print(df)


# In[5]:


import pandas as pd
data=[['ali', 45], ['raza', 40], ['qaim', 6]]
df=pd.DataFrame(data,columns=['name', 'age'])
print(df)


# In[9]:


with open('label.lst') as file:
  lines=file.readlines()


# In[10]:


lines


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




