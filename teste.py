composicao = [[0.0252, 0.0581, 0.0754, 0.1028, 0.1257, 0.1564, 0.1776, 0.2062, 0.2231, 0.2487, 0.2761, 0.3084, 0.3248, 0.3509, 0.3753, 0.4025, 0.4274],
              [0.0252, 0.0581, 0.0754, 0.1028, 0.1257, 0.1564, 0.1776, 0.2062, 0.2231, 0.2487, 0.2761, 0.3084, 0.3248, 0.3509, 0.3753, 0.4025, 0.4274],
              [0.0252, 0.0581, 0.0754, 0.1028, 0.1257, 0.1564, 0.1776, 0.2062, 0.2231, 0.2487, 0.2761, 0.3084, 0.3248, 0.3509, 0.3753, 0.4025, 0.4274],
              [0.0252, 0.0581, 0.0754, 0.1028, 0.1257, 0.1564, 0.1776, 0.2062, 0.2231, 0.2487, 0.2761, 0.3084, 0.3248, 0.3509, 0.3753, 0.4025, 0.4274],
              [0.0252, 0.0581, 0.0754, 0.1028, 0.1257, 0.1564, 0.1776, 0.2062, 0.2231, 0.2487, 0.2761, 0.3084, 0.3248, 0.3509, 0.3753, 0.4025, 0.4274]]

temperatura = [[298.15, 298.11, 298.13, 298.09, 298.1, 298.11, 298.1, 298.09, 298.14, 298.11, 298.12, 298.11, 298.1, 298.13, 298.1, 298.13, 298.17],
                [303.14, 303.12, 303.12, 303.12, 303.16, 303.08, 303.12, 303.12, 303.16, 303.13, 303.08, 303.15, 303.16, 303.11, 303.09, 303.1, 303.12],
                [313.12, 313.15, 313.1, 313.11, 313.12, 313.13, 313.11, 313.1, 313.18, 313.09, 313.16, 313.13, 313.14, 313.1, 313.17, 313.08, 313.11],
                [323.13, 323.11, 323.14, 323.1, 323.1, 323.16, 323.1, 323.07, 323.15, 323.16, 323.17, 323.13, 323.17, 323.12, 323.11, 323.1, 323.16],
                [333.12, 333.1, 333.12, 333.12, 333.08, 333.12, 333.14, 333.11, 333.1, 333.11, 333.09, 333.11, 333.08, 333.17, 333.13, 333.09, 333.12]]

pressao = [[0.24, 0.67, 0.9, 1.31, 1.67, 2.19, 2.56, 3.06, 3.34, 3.72, 4.1, 4.53, 4.74, 5.06, 5.34, 5.63, 5.89],
            [0.33, 0.8, 1.06, 1.48, 1.92, 2.49, 2.84, 3.33, 3.61, 4.05, 4.47, 4.87, 5.08, 5.37, 5.69, 6.02, 6.32],
            [0.42, 0.96, 1.26, 1.74, 2.2, 2.8, 3.21, 3.77, 4.03, 4.45, 4.88, 5.32, 5.52, 5.87, 6.1, 6.43, 6.71],
            [0.55, 1.14, 1.49, 2.06, 2.54, 3.17, 3.61, 4.15, 4.46, 4.91, 5.35, 5.83, 6.06, 6.39, 6.68, 6.98, 7.25],
            [0.62, 1.43, 1.86, 2.52, 3.05, 3.74, 4.23, 4.78, 5.1, 5.56, 6.03, 6.53, 6.77, 7.13, 7.45, 7.79, 8.09]]

Y_predito = []
loss = []
precisao = []

for j in range(len(composicao)):
     X = np.array([composicao[j], temperatura[j],
            [197.97 for i in range(17)],
            [2.36 for i in range(17)],
            [596.21 for i in range(17)],
            [0.8087 for i in range(17)],
            [540.77 for i in range(17)],
            [449.52 for i in range(17)],
            [5.983 for i in range(17)],
            [5.983 for i in range(17)],
            [24.263 for i in range(17)],
            [1.932 for i in range(17)],
            [179.321 for i in range(17)],
            [0 for i in range(17)],
            [0 for i in range(17)],
            [162.874 for i in range(17)],
            [70.129 for i in range(17)],
            [107.14 for i in range(17)],
            [0.908 for i in range(17)],
            [0.449 for i in range(17)],
            [6 for i in range(17)],
            [0.673 for i in range(17)],
            [0 for i in range(17)],
            [0 for i in range(17)],
            [-0.71 for i in range(17)],
            [8.81 for i in range(17)]])

     Y = np.array([pressao[j],
          [1 for i in range(17)]])

     Y_f = (Y - np.min(Y))/ (np.max(Y) - np.min(Y))

     Y_f[1,:] = 1

     rede = NeuralNetwork([
                        Input(26),\
                        Hidden(260),\
                        Output(2)
                    ])
     rede.train(X=X, y=Y_f, epochs=100000, learning_rate=10e-5, function_activate='tanh', optimizer='Adam')

     Y_calc = rede.predict(X)
     Y_predito.append(Y_calc)
     loss.append(rede.loss)
     precisao.append(rede.precision)

fig = go.Figure()

# Adicionando traços ao gráfico
fig.add_trace(go.Scatter(x=[i for i in range(100000)], y=rede.loss, name='Loss'))
fig.add_trace(go.Scatter(x=[i for i in range(100000)], y=[0 if i <0 else i for i in rede.precision], name='Precisão'))

# Adicionando rótulos aos eixos e ao layout geral
fig.update_layout(
    xaxis_title='Épocas',
    yaxis_title='Valor',
)

# Mostrando o gráfico
fig.show()

fig = go.Figure()

# Adicionando traços ao gráfico
fig.add_trace(go.Scatter(x=X[0], y=Y_calc[0], name='Predito', mode='lines'))
fig.add_trace(go.Scatter(x=X[0], y=Y_f[0], name='Real', mode='markers'))

# Adicionando rótulos aos eixos e ao layout geral
fig.update_layout(
    xaxis_title='X',
    yaxis_title='Y',
)

# Mostrando o gráfico
fig.show()

fig = go.Figure()
for i in range(5):
    fig.add_trace(go.Scatter3d(x=composicao[i], y=Y_predito[i][0], z=[np.mean(temperatura[i])] * len(X[0]), name='Predito', mode='lines'))
    fig.add_trace(go.Scatter3d(x=composicao[i], y=(pressao[i] - np.min(pressao[i]))/ (np.max(pressao[i])-np.min(pressao[i])), z=[np.mean(temperatura[i])] * len(X[0]), name='Real', mode='markers'))

    # Adicionando rótulos aos eixos e ao layout geral
    fig.update_layout(
        scene=dict(
            xaxis_title='Composição',
            yaxis_title='Pressão',
            zaxis_title='Temperatura',
        ),
        width=1000,  # ajuste o valor conforme necessário
        height=800,  # ajuste o valor conforme necessário,
        title = '[emim][BF4]'
    )

    # Mostrando o gráfico
fig.show()