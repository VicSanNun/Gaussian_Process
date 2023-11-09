# Abrindo o arquivo de texto para leitura
with open('outupt_199.txt', 'r') as file:
    lines = file.readlines()

# Processando os valores da coluna 'z' e atualizando para o novo intervalo
new_lines = []
for line in lines:
    data = line.split()
    if data:  # Verifica se há dados na linha
        z_value = int(data[0])
        new_z = z_value * (2.5 / 199)  # Convertendo o valor de z para o novo intervalo
        data[0] = str(new_z)
        new_line = '\t'.join(data)
        new_lines.append(new_line + '\n')

# Salvando os dados modificados em um novo arquivo
with open('dados_modificados.txt', 'w') as new_file:
    new_file.writelines(new_lines)
