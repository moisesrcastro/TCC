import numpy as np
from sklearn.metrics import r2_score
class NeuralNetwork():

    def __init__(self, layers: list):

        self.layers = layers

    def __relu(self, x):
        return np.maximum(0, x)
    
    def __relu_derivative(self, x):
        result = np.zeros_like(x)
        result[x >= 0] = 1
        return result
    
    def __linear(self, x):
        return x
    
    def __linear_derivate(self, x):
        return np.ones(x.shape)
    
    def __sigmoide(self, x):
        return  1 / (1 + np.exp(-x))
    
    def __sigmoide_derivative(self, x):
        return self.__sigmoide(x) * (1 - self.__sigmoide(x))
    
    def __tanh(self, x):
        return  np.tanh(x)
    
    def __tanh_derivative(self, x):
        return 1 - self.__tanh(x) ** 2
    
    def __elu(self, x, alpha=1.0):
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    def __elu_derivative(self, x, alpha=1.0):
        return np.where(x >= 0, 1, self.__elu(x, alpha) + alpha)
    
    def __leaky_relu(self, x, alpha=0.01):
        return np.where(x >= 0, x, alpha * x)

    def __leaky_relu_derivative(self, x, alpha=0.01):
        return np.where(x >= 0, 1, alpha)
    
    
    def train(self, X, y, learning_rate=10e-2, epochs=1000, function_activate='sigmoide', optimizer='gradient-descent'):

        self.function_activate = function_activate

        if function_activate == 'sigmoide':
            function = self.__sigmoide
            derivate = self.__sigmoide_derivative

        elif function_activate == 'tanh':
            function = self.__tanh
            derivate = self.__tanh_derivative

        elif function_activate == 'linear':
            function = self.__linear
            derivate = self.__linear_derivate

        elif function_activate == 'ReLu':
            function = self.__relu
            derivate = self.__relu_derivative

        elif function_activate == 'eLu':
            function = self.__elu
            derivate = self.__elu_derivative
    
        elif function_activate == 'LeakyReLu':
            function = self.__leaky_relu
            derivate = self.__leaky_relu_derivative


        self.function = function
        self.derivate = derivate
        
        mse = []
        coef_deter = []

        #---------------> INICIALIZAÇÃO DOS PESOS, VIESES, MOMENTUMS E VELOCIDADE
        for i in range(1, len(self.layers)):

            globals()['W_' + str(i)] = np.random.normal(loc=0, scale=1, size=(self.layers[i].n, self.layers[i-1].n))
            globals()['Mw_' + str(i)] = np.zeros((self.layers[i].n, self.layers[i-1].n))
            globals()['Vw_' + str(i)] = np.zeros((self.layers[i].n, self.layers[i-1].n))

            globals()['B_' + str(i)] = np.random.normal(loc=0, scale=1, size=(self.layers[i].n, 1))
            globals()['Mb_' + str(i)] = np.zeros((self.layers[i].n, 1))
            globals()['Vb_' + str(i)] = np.zeros((self.layers[i].n, 1))
        
        uni = np.ones((self.layers[-1].n, 1))

        for m in range(epochs):
            Rede = []
            #---------------> CRIANDO A ESTRUTURA DA REDE
            for j in range(len(self.layers)):
                if j == 0:
                    Rede.append([[0, 0], X])

                else:
                    Rede.append([[eval('W_'+ str(j)), eval('B_'+ str(j))],\
                                            function(np.dot(eval('W_'+ str(j)), Rede[j-1][1])+ eval('B_'+ str(j)))])
                    
            #--------------->CALCULANDO O ERRO
            Erro = (Rede[-1][1] - y)

            pesos = ['fator_W_' + str(k) for k in range(len(Rede)-1, 0, -1)]
            bias = ['fator_B_' + str(k) for k in range(len(Rede)-1, 0, -1)]
            
            
            for l in range(1, len(Rede)):
                if l == 1:
                    globals()[pesos[l-1]] = np.dot(np.multiply(Erro, derivate(Rede[-l][1])), function(Rede[-(l+1)][1]).transpose())
                else:
                
                    fator = Erro

                    for n in range(1, l):

                        fator = np.dot((fator * derivate(Rede[-n][1])).transpose(), Rede[-n][0][0]).transpose()
                    
                    globals()[pesos[l-1]] = np.dot(fator * derivate(Rede[-l][1]), function(Rede[-(l+1)][1]).transpose())

            for i in range(1, len(Rede)):
                if i == 1:
                    globals()[bias[i-1]] = np.dot(np.dot(Erro, derivate(Rede[-i][1].transpose())), uni)

                elif i == 2:
                    globals()[bias[i-1]] = np.dot(np.dot(derivate(Rede[-i][1]), (Erro * derivate(Rede[-(i-1)][1])).transpose()) * Rede[-1][0][0].transpose(), uni)

                else:
                    fator_bias = np.dot(Rede[-(i-1)][0][0], derivate(Rede[-(i)][1]))
                    for k in range(1, i-1):
                        
                        globals()[bias[i-1]] = np.dot(np.dot(fator_bias * derivate(Rede[-2][1]), (Erro * Rede[-1][1]).transpose()) * Rede[-1][0][0].transpose(), uni)
                        
                        if (k+i) <= len(Rede):
                            fator_bias *= np.dot(Rede[-(i-1+k)][0][0], derivate(Rede[-(i+k)][1]))
                            
                        else: pass

            if optimizer == 'gradient-descent':
                for i in range(1, len(Rede)):
                    
                    globals()['W_'+str(i)] -= learning_rate * eval(pesos[-i])
                    globals()['B_'+str(i)] -= learning_rate * eval(bias[-i])        

            elif optimizer == 'Adam':
                B1 = 0.9
                B2 = 0.99
                epsilon = 10e-6

                for i in range(1, len(Rede)):
                    #MOMENTUNS
                    globals()['Mw_' + str(i)] = B1 * eval('Mw_' + str(i)) + (1 - B1) * eval(pesos[-i])
                    globals()['Mb_' + str(i)] = B1 * eval('Mb_' + str(i)) + (1 - B1) * eval(bias[-i])

                    #VELOCIDADES
                    globals()['Vw_' + str(i)] = B2 * eval('Vw_' + str(i)) + (1 - B2) * np.power(eval(pesos[-i]), 2)
                    globals()['Vb_' + str(i)] = B2 * eval('Vb_' + str(i)) + (1 - B2) * np.power(eval(bias[-i]), 2)

                    globals()['W_'+str(i)] -= learning_rate * globals()['Mw_' + str(i)] / (np.sqrt(eval('Vw_' + str(i))) + epsilon)
                    globals()['B_'+str(i)] -= learning_rate * globals()['Mb_' + str(i)] / (np.sqrt(eval('Vb_' + str(i))) + epsilon)
            


            mse.append(np.sum(Erro ** 2)/len(y))
            coef_deter.append(r2_score(Rede[-1][1], y))
        
        self.pesos = []
        self.vieses = []

        for i in range(1, len(self.layers)):

            self.pesos.append(eval('W_' + str(i)))
            self.vieses.append(eval('B_' + str(i)))

        self.loss = mse
        self.precision = coef_deter
        
        return 
    
    def predict(self, x):


        for i  in range(1, len(self.layers)):
            if i == 1: 
                y_calc = self.function(np.dot(self.pesos[i-1], x) + self.vieses[i-1])

            else:

                y_calc =  y_calc = self.function(np.dot(self.pesos[i-1], y_calc) + self.vieses[i-1])

        return y_calc
