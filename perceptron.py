# Perceptron sem bias para a porta AND

# Tabela verdade AND
entradas = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

saidas_esperadas = [0, 0, 0, 1]

# Pesos iniciais
w1 = 0.5
w2 = 0.5

# Taxa de aprendizagem
taxa = 0.1

# Função de ativação
def ativacao(soma):
    if soma >= 1:
        return 1
    else:
        return 0

# Treinamento
for epoca in range(10):

    print(f"\nÉpoca {epoca+1}")

    for i in range(len(entradas)):

        x1 = entradas[i][0]
        x2 = entradas[i][1]

        soma = (x1 * w1) + (x2 * w2)

        saida = ativacao(soma)

        erro = saidas_esperadas[i] - saida

        # Atualização dos pesos
        w1 = w1 + taxa * erro * x1
        w2 = w2 + taxa * erro * x2

        print(f"Entrada: {x1}, {x2}")
        print(f"Saída esperada: {saidas_esperadas[i]}")
        print(f"Saída obtida: {saida}")
        print(f"Erro: {erro}")
        print(f"Novos pesos: w1={w1:.2f}, w2={w2:.2f}")

# Teste final
print("\nTeste Final")

for entrada in entradas:

    soma = (entrada[0] * w1) + (entrada[1] * w2)

    saida = ativacao(soma)

    print(f"Entrada: {entrada} -> Saída: {saida}")