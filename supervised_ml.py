import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Prepare the training dataset
# hours studied
X = [[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

# For examination_score
Y = [30, 40, 45, 55, 65, 70, 80, 85, 90, 95]

# Create the Linear Regression model
model = LinearRegression()
# Train the model using fit
model.fit(X,Y)

# user to enter the number of hours studied as input
print(" Linear Regression for Supervised ML ")
entery = input("Enter hours studied: ")
hours = float(entery)
# Using prediction to estimate the exam score
prediction = model.predict([[hours]])


# Display the prediction rounded to two decimal places
print("Predicted exam score: ", round(prediction[0],2))

#print(f"Estimated examination score: {prediction:.2f}")

# Plot the training data points
plt.scatter(X,Y, color="blue")

# Plots the regression line
plt.plot(X, model.predict(X), color="green")
    

# labels and displaying the graph
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Predicted Exam Score")
plt.show()
