import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def read_data(file_path):
    data = np.genfromtxt(file_path)
    z = data[:, 0].reshape(-1, 1)
    hz = data[:, 1]
    return z, hz

def linear_regression(z, hz):
    model = LinearRegression()
    model.fit(z, hz)
    return model.coef_[0], model.intercept_

def polynomial_regression(z, hz, degree=2):
    coeffs = np.polyfit(z.flatten(), hz, degree)
    return coeffs

def plot_regression(z, hz, slope, intercept, poly_coeffs):
    plt.scatter(z, hz, label='Dados', color='b')
    plt.plot(z, slope * z + intercept, color='r', label='Regressão Linear')
    
    # Gerar valores de z para o plot da regressão polinomial
    z_values = np.linspace(min(z), max(z), 100)
    hz_poly = np.polyval(poly_coeffs, z_values)
    plt.plot(z_values, hz_poly, color='g', label='Regressão Polinomial (grau 2)')

    plt.xlabel('Z')
    plt.ylabel('H(z)')
    plt.legend()
    plt.title('Regressões Linear e Polinomial em z')
    plt.grid(True)
    plt.show()

def main():
    file_path = 'inputdata.txt'
    z, hz = read_data(file_path)

    slope, intercept = linear_regression(z, hz)
    poly_coeffs = polynomial_regression(z, hz, degree=2)

    print("Regressão Linear em z:")
    print(f"Coeficiente (slope): {slope}")
    print(f"Intercepto: {intercept}")

    print("Regressão Polinomial (grau 2) em z:")
    print(f"Coeficientes do polinômio: {poly_coeffs}")

    plot_regression(z, hz, slope, intercept, poly_coeffs)

if __name__ == "__main__":
    main()
