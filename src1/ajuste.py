# -*- coding: utf-8 -*-
"""
Python 3
09 / 07 / 2024
@author: OzzyLoachamin

"""

import numpy as np
        
#----------------Ajuste de mínimos cuadrados---------------

def ajusteMinimosCuadrados(n, grado, xi, yi):
    A = np.zeros((grado+1,grado+1))
    b = np.zeros((grado+1,1))
    for i in range (0,grado+1):
        for j in range(0,grado+1):
            k=0
            while (k<n):
                A[i,j] += xi[k]**((i)+(j))
                k += 1
        for l in range(n):
            b[i] += (xi[l]**(i))*yi[l]
            
    imprimirMatriz(A, "Matriz A")
    imprimirMatriz(b, "Vector b")
    return A,b

def hallarCoeficientes(a,b):
    A=np.linalg.inv(a)
    x = np.dot(A,b)
    imprimirMatriz(x, "Coeficientes del polinomio")
    return x

#--------------------------Graficar-------------------------

import matplotlib.pyplot as plt
import sympy as sym

def graficarAjustePolinomial(xi, yi, c, colorcurva, rango_x, rango_y, x_pol, y_pol, lim_inf):
    x = sym.Symbol('x')
    f_x = sum(round(coef[0], 4) * x**i for i, coef in enumerate(c))
    
    # Generar valores de x
    x_val = np.linspace(min(xi) - lim_inf, max(xi) + lim_inf, 1000)
    f = sym.lambdify(x, f_x, modules=['numpy'])
    y = f(x_val)
    
    xi = np.array(xi)
    yi = np.array(yi)
    
    # Calcular los residuos (errores)
    residuos = yi - f(xi)
    imprimirErrores(residuos,xi)
    
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean(residuos**2)
    print("El error cuadrático medio para este ajuste es de:", round(mse,6))
    imprimirPolinomio(f_x)
    
    # Graficar
    plt.figure(figsize=(10, 8))
    
    # Graficar la curva experimental en color morado
    plt.plot(x_val, y, color=colorcurva, linestyle='-', linewidth=2, label='Curva Ajustada')
    
    
    # Graficar los datos originales como puntos negros
    plt.scatter(xi, yi, color='black', label='Puntos Originales', s=30, marker='o')
    # Graficar los puntos originales con barras de error
    plt.errorbar(xi, yi, yerr=abs(residuos), fmt=' ', color='red', markersize=5, capsize=5, capthick=2, label='Puntos con Error')
    
    # Etiquetas y límites de los ejes
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xlim(rango_x)
    plt.ylim(rango_y)
    
    # Título
    plt.title('Ajuste Polinomial', fontsize=15)
    
    # Cuadrícula
    plt.grid(True)
    
    # Leyenda
    plt.legend(fontsize=12)
    
    # Texto con la ecuación del polinomio
    plt.text(x_pol, y_pol, f'$P(x) = {sym.latex(f_x)}$', fontsize=14, color=colorcurva, verticalalignment='bottom')
    
    # Estilo de los ejes
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Mostrar el gráfico
    plt.show()
#------------------------------Graficar no lineales-------------------------------

def graficarAjusteNoLineal(xi, yi, f_x, colorcurva, rango_x, rango_y, x_pol, y_pol, lim_inf):   
    x = sym.Symbol('x')
    
    # Generar valores de x
    x_val = np.linspace(min(xi) - lim_inf, max(xi) + lim_inf, 100)
    f = sym.lambdify(x, f_x, modules=['numpy'])
    y = f(x_val)
    
    xi = np.array(xi)
    yi = np.array(yi)
    
    # Calcular los residuos (errores)
    residuos = yi - f(xi)
    imprimirErrores(residuos,xi)
    
    # Calcular el error cuadrático medio (MSE)
    mse = np.mean(residuos**2)
    print("El error cuadrático medio para este ajuste es de:", round(mse,2))
    imprimirPolinomio(f_x)
    
    # Graficar
    plt.figure(figsize=(10, 8))
    
    # Graficar la curva experimental
    plt.plot(x_val, y, color=colorcurva, linestyle='-', linewidth=2, label='Curva Ajustada')
    
    # Graficar los puntos originales con barras de error estilo I mayúscula
    # plt.errorbar(xi, yi, yerr=[residuos, residuos], fmt=' ', uplims=True, lolims=True, capsize=5, capthick=2, color='red', label='Puntos con Error')
    plt.errorbar(xi, yi, yerr=abs(residuos), fmt=' ', color='red', markersize=5, capsize=5, capthick=2, label='Puntos con Error')

    # Graficar los datos originales como puntos negros
    plt.scatter(xi, yi, color='black', label='Puntos Originales', s=30, marker='o')
    
    # Etiquetas y límites de los ejes
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.xlim(rango_x)
    plt.ylim(rango_y)
    
    # Título
    plt.title('Ajuste No Lineal', fontsize=15)
    
    # Cuadrícula
    plt.grid(True)
    
    # Leyenda
    plt.legend(fontsize=12)
    
    # Texto con la ecuación del polinomio
    plt.text(x_pol, y_pol, f'$P(x) = {sym.latex(f_x)}$', fontsize=14, color=colorcurva, verticalalignment='bottom')
    
    # Estilo de los ejes
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Mostrar el gráfico
    plt.show()
#--------------------------Regreso a la expresión original-----------------------
from IPython.display import display, Math

def calcularExpresionOriginal(c,exp):
    print("Con los coeficientes asociados al polinomio linealizado hallamos los coeficientes de nuestra\n expresión:\n")  
    #Hallar los coeficientes adecuados
    b_exp = np.e**(c[0,0])
    a_exp = c[1,0]
    print("a =",a_exp," y b =", b_exp,"\n")
        
    x = sym.Symbol('x')
    if exp:
        #Generar la ecuación en la forma be^{ax}
        f_x = round(b_exp,4)*sym.exp(round(a_exp,4)*x)
    else:
        #Generar la ecuación en la forma bx^{a}
        f_x = round(b_exp,4)*x**(round(a_exp,4))
    return f_x

#-----------------------Impresión--------------------------

def imprimirMatriz(matrix, name):
    print(name + ":")
    for row in matrix:
        print(" [ ", "   ".join(f"{elem:12.4f}" for elem in row), "]")
        
#---------------------------Imprimir errores puntuales----------------------------
def imprimirErrores(residuos,xi):
    i=1
    print(" ")
    for res in residuos:
        print("El error absoluto de f(x_"+str(i)+") al punto x_"+str(i)+" es de",round(abs(res),6))
        i += 1
        
#------------------------------Impresión polinomios-----------------------------    
def imprimirPolinomio(f_x):    
    # Generar la representación LaTeX de la expresión
    latex_expr = sym.latex(f_x)
    print("Por tanto, el polinomio aproximado en la forma solicitada es:\n")
    # Mostrar la expresión LaTeX
    display(Math(latex_expr))