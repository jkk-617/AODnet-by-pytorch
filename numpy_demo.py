import numpy as np
a = np.zeros(10)
print(a.size)
print(a)

a = np.zeros((14,))
print(a.size)
print(a)

a = np.arange(4)
print(a)

a = np.random.random_sample(4)
print(a.size)
print(a)

a = np.random.rand(10)
print(a.size)
print(a)

a = np.array([1, 3, 4, 5])
print(a.size)
print(a)
print(a[-1])


a = np.arange(10)
print(a[1:10:3])
print(a[2:])
print(a[3:9])

print(-a)
print(np.sum(a))
print(np.mean(a))
print(a**2)

b = np.arange(2,12)
print(b)
print(a + b)
print(np.dot(a,b))

print(np.zeros([3,4]))

print(np.zeros([3,4,5]))




















