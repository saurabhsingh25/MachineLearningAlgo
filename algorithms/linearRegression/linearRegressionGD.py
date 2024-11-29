import numpy as np

#Y -> (1,m), X-> (n,m), w->(n,1)
#Y= wT*X
#yhat= w0+ w1x1+ w2x2.....

class LinearRegression:
    def __init__(self):
        self.learning_rate= 0.001
        self.iterations= 30000

    def y_hat(self,X,w):
        return np.dot(w.T,X)

    def loss(self,y_hat,y):
        L= 1/self.m * np.sum(np.power(y_hat-y,2))
        return L

    def gradient_descent(self,w,X,y,y_hat):
        dLdW=2/self.m*np.dot(X,(y_hat-y).T)
        w=w-self.learning_rate*dLdW
        return w

    def main(self,X,y):
        ones=np.ones((1,X.shape[1]))
        X= np.append(X,ones,axis=0)

        self.m= X.shape[1]
        self.n= X.shape[0]

        w=np.zeros((self.n,1))

        for i in range(self.iterations+1):
            yhat=self.y_hat(X, w)
            loss=self.loss(yhat,y)

            if i%2000==0:
                print(f'Loss at {i} iteration is {loss}')
            w=self.gradient_descent(w,X,y,yhat)

        return w

if __name__ == "__main__":
    X= np.random.rand(1,500)
    y= 3*X + np.random.rand(1,500)*0.01
    regression= LinearRegression()
    w= regression.main(X,y)
