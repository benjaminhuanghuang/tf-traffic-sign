# Import `tensorflow`
import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Print the result
print("multipy of x1 and x2:")
print(result)


# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print("session run multipy of x1 and x2")
print(sess.run(result))

# Close the session
sess.close()

'''
with tf.Session() as sess:
    output = sess.run(result)
    print(output)
'''