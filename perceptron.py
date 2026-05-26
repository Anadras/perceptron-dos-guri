import numpy as np

# Entradas
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Saídas esperadas (AND)
y = np.array([0, 0, 0, 1])

# Inicialização dos pesos
pesos = np.zeros(2)

# Bias
bias = 0

# Taxa de aprendizado
taxa = 0.1

# Função de ativação
def ativacao(soma):
    if soma >= 0:
        return 1
    return 0

# Treinamento
for epoca in range(10):

    print(f"\nÉpoca {epoca + 1}")

    for i in range(len(X)):

        # Soma ponderada
        soma = np.dot(X[i], pesos) + bias

        # Previsão
        y_pred = ativacao(soma)

        # Erro
        erro = y[i] - y_pred

        # Atualização dos pesos
        pesos += taxa * erro * X[i]

        # Atualização do bias
        bias += taxa * erro

        print(f"Entrada: {X[i]}")
        print(f"Saída prevista: {y_pred}")
        print(f"Erro: {erro}")
        print(f"Pesos: {pesos}")
        print(f"Bias: {bias}")

print("\nTreinamento finalizado!")

# Teste final
for i in range(len(X)):
    soma = np.dot(X[i], pesos) + bias
    y_pred = ativacao(soma)

    print(f"{X[i]} -> {y_pred}")