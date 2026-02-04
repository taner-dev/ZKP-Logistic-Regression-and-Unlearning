import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))


#Cross Entropy wurde hier benutzt aber mit festen grenzen und nicht ins unendliche
def KostenFunktion(X, y, w, b):
     z=np.dot(X, w)+b
     g=sigmoid(z)
     eps=1e-15
     ng=np.clip(g, eps, 1-eps)
     return -np.mean(y*np.log(ng)+(1-y)*np.log(1-ng))


def GradientenFunktion(X, y, w, b):
    m=len(X)
    z=np.dot(X, w)+b
    g=sigmoid(z)
    fehler=g-y
    grad_w=np.dot(X.T, fehler)/m
    grad_b=np.sum(fehler)/m
    return grad_w,grad_b

#Funktion, um später Vorhersagen zu tätigen
def predict(X, w, b):
    z=np.dot(X, w)+b
    return (sigmoid(z)>=0.5).astype(int)

#Die "main" methode, um w und b überhaupt zu berechnen
def Gradientenabstieg(X, y, alpha=0.05, iterations=10000): #Hier kann man Grad der Änderung und die Iterationen anpassen
 n=X.shape[1]
 w=np.zeros(n)
 b=0
 for i in range(iterations):
    grad_w, grad_b=GradientenFunktion(X, y, w, b)
    w-=alpha*grad_w
    b-=alpha*grad_b
    if i % 1000 == 0:
     print(f"Iteration {i}: Kosten {KostenFunktion(X, y, w, b)}")


 predictions = predict(X, w, b)
 accuracy = np.mean(predictions==y) * 100
 print(f"Training accuracy: {accuracy:.2f}%") 
      
 return w,b


