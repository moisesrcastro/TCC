#------------> CONFIGURAÇÕES DA CAMADA DE ENTRADA
class Input():
    
    def __init__(self, n):

        self.n = n

#------------> CONFIGURAÇÕES DAS CAMADAS OCULTAS
class Hidden():

    def __init__(self, n, media=0, desvio=1):

        self.n = n
        self.media = media
        self.desvio = desvio

#------------> CONFIGURAÇÕES DA CAMADA DE SAÍDA
class Output():
    
    def __init__(self, n, media=0, desvio=1):

        self.n = n
        self.media = media
        self.desvio = desvio