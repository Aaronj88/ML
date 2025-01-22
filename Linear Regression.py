import numpy as np
import matplotlib.pyplot as plt

rand = np.random.randint(5,10,10)
values_x = np.arange(0,10)
values_y = 2*values_x+rand

plt.scatter(values_x,values_y)
plt.show()

#m = sum((xi-mean(x)) * (yi-mean(y))) / sum((xi – mean(x))^2)
#c = mean(y) – m * mean(x)

mean_x = values_x.mean()
mean_y= values_y.mean()

m = np.sum((values_x-mean_x)*(values_y-mean_y))/np.sum((values_x-mean_x)**2)
c = mean_y - m * mean_x

print("slope(m): ",m,"y - intercept(c): ",c)
py = m*values_x + c

plt.scatter(values_x,values_y)
plt.plot(values_x,py,"r-o")
plt.show()

#RMSE - Root Mean Squared Error sqrt( sum( (p – yi)^2 )/n )

rmse = np.sqrt(((py-values_y)**2).mean())
print("RMSE: ",rmse)

#simple linear regression
from sklearn.linear_model import LinearRegression
X_2d = values_x.reshape(-1,1)

model = LinearRegression()
model.fit(X_2d,values_y)

print(model.coef_,model.intercept_)

predictions = model.predict(X_2d)
print(predictions,py)


from sklearn.metrics import root_mean_squared_error
err = root_mean_squared_error(values_y,predictions)
print(err,"ours:",rmse)




