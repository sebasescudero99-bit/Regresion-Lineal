/*
 * Universidad EAFIT - Escuela de Ciencias Aplicadas e Ingeniería
 * Ingeniería de Sistemas - Lenguajes y Paradigmas de Programación
 * Práctica No. 2: Programación Orientada a Objetos (Regresión Lineal Múltiple y Análisis de Memoria)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <iomanip>

using namespace std;




class LinearRegression {
private:
    vector<double> weights; // Arreglo con los pesos del modelo (w1, w2, ..., wn)
    double bias;            // Término de sesgo o intercepto (b)

public:
    // Constructor por defecto
    LinearRegression() {
        bias = 0.0;
    }

    // Constructor parametrizado con el número de variables de entrada
    LinearRegression(int num_variables) {
        weights.resize(num_variables, 0.0);
        bias = 0.0;
    }





    // Calcula la media aritmética (promedio) de un vector de datos
    double calcular_media(const vector<double>& datos) {
        if (datos.empty()) return 0.0;
        double suma = 0.0;
        for (size_t i = 0; i < datos.size(); i++) {
            suma += datos[i];
        }
        return suma / datos.size();
    }

    // Calcula la desviación estándar poblacional de un vector de datos
    double calcular_desviacion_estandar(const vector<double>& datos, double media) {
        if (datos.empty()) return 1.0;
        double suma_cuadrados = 0.0;
        for (size_t i = 0; i < datos.size(); i++) {
            double diferencia = datos[i] - media;
            suma_cuadrados += diferencia * diferencia;
        }
        double desviacion = sqrt(suma_cuadrados / datos.size());
        if (desviacion == 0.0) return 1.0; // Evita división por cero
        return desviacion;
    }





    // Calcula la matriz transpuesta de una matriz dada (filas pasan a columnas)
    vector<vector<double>> transponer_matriz(const vector<vector<double>>& matriz) {
        if (matriz.empty()) return {};
        size_t filas = matriz.size();
        size_t columnas = matriz[0].size();
        vector<vector<double>> transpuesta(columnas, vector<double>(filas, 0.0));

        for (size_t i = 0; i < filas; i++) {
            for (size_t j = 0; j < columnas; j++) {
                transpuesta[j][i] = matriz[i][j];
            }
        }
        return transpuesta;
    }

    // Multiplica una matriz por un vector: resultado = matriz * vec
    vector<double> multiplicar_matriz_vector(const vector<vector<double>>& matriz, const vector<double>& vec) {
        size_t filas = matriz.size();
        if (filas == 0) return {};
        size_t columnas = matriz[0].size();
        vector<double> resultado(filas, 0.0);

        for (size_t i = 0; i < filas; i++) {
            double suma = 0.0;
            for (size_t j = 0; j < columnas; j++) {
                suma += matriz[i][j] * vec[j];
            }
            resultado[i] = suma;
        }
        return resultado;
    }

    



    // data_scaling(): Escala los valores del dataset usando estandarización (Z-score)
    void data_scaling(vector<vector<double>>& X) {
        if (X.empty() || X[0].empty()) return;
        size_t filas = X.size();
        size_t columnas = X[0].size();

        // Transponemos temporalmente para extraer cada columna fácilmente
        vector<vector<double>> X_T = transponer_matriz(X);

        for (size_t j = 0; j < columnas; j++) {
            double media = calcular_media(X_T[j]);
            double desviacion = calcular_desviacion_estandar(X_T[j], media);

            for (size_t i = 0; i < filas; i++) {
                X[i][j] = (X[i][j] - media) / desviacion;
            }
        }
    }

    // fit(): Entrena el modelo de regresión lineal mediante descenso de gradiente
    void fit(const vector<vector<double>>& X_train, const vector<double>& y_train, double tasa_aprendizaje = 0.05, int epocas = 2000) {
        if (X_train.empty() || y_train.empty()) return;
        size_t n_muestras = X_train.size();
        size_t n_variables = X_train[0].size();

        // Inicializar pesos y sesgo en cero
        weights.assign(n_variables, 0.0);
        bias = 0.0;

        // Precalcular la matriz transpuesta de X_train para el cálculo de gradientes
        vector<vector<double>> X_train_T = transponer_matriz(X_train);

        for (int epoca = 0; epoca < epocas; epoca++) {
            // 1. Calcular las predicciones actuales: y_hat = X_train * weights + bias
            vector<double> y_hat = predict(X_train);

            // 2. Calcular el vector de errores: error = y_hat - y_train
            vector<double> error(n_muestras, 0.0);
            double suma_error = 0.0;
            for (size_t i = 0; i < n_muestras; i++) {
                error[i] = y_hat[i] - y_train[i];
                suma_error += error[i];
            }

            // 3. Calcular gradientes de los pesos: grad_w = (2/N) * (X_train^T * error)
            vector<double> grad_w = multiplicar_matriz_vector(X_train_T, error);

            // 4. Actualizar pesos (weights) y sesgo (bias)
            for (size_t j = 0; j < n_variables; j++) {
                weights[j] -= tasa_aprendizaje * (2.0 / n_muestras) * grad_w[j];
            }
            bias -= tasa_aprendizaje * (2.0 / n_muestras) * suma_error;
        }
    }

    // predict(): Predice las salidas para un conjunto de datos de entrada X
    vector<double> predict(const vector<vector<double>>& X) {
        vector<double> y_hat = multiplicar_matriz_vector(X, weights);
        for (size_t i = 0; i < y_hat.size(); i++) {
            y_hat[i] += bias;
        }
        return y_hat;
    }

    // score(): Calcula el error de predicción mediante el Error Cuadrático Medio (MSE)
    double score(const vector<double>& y_real, const vector<double>& y_hat) {
        if (y_real.empty() || y_real.size() != y_hat.size()) return 0.0;
        double suma_error_cuadratico = 0.0;
        for (size_t i = 0; i < y_real.size(); i++) {
            double diff = y_real[i] - y_hat[i];
            suma_error_cuadratico += diff * diff;
        }
        return suma_error_cuadratico / y_real.size();
    }

    // Calcula el coeficiente de determinación R^2 como métrica complementaria de evaluación
    double calcular_r2(const vector<double>& y_real, const vector<double>& y_hat) {
        if (y_real.empty() || y_real.size() != y_hat.size()) return 0.0;
        double media_y = calcular_media(y_real);
        double ss_res = 0.0;
        double ss_tot = 0.0;
        for (size_t i = 0; i < y_real.size(); i++) {
            double res = y_real[i] - y_hat[i];
            double tot = y_real[i] - media_y;
            ss_res += res * res;
            ss_tot += tot * tot;
        }
        if (ss_tot == 0.0) return 0.0;
        return 1.0 - (ss_res / ss_tot);
    }

    // Métodos Getter solicitados en el pseudocódigo de la guía
    vector<double> getWeigths() const {
        return weights;
    }

    vector<double> getWeights() const {
        return weights;
    }

    double getBias() const {
        return bias;
    }
};




LinearRegression modelo_global;




bool cargar_csv(const string& ruta_archivo, vector<vector<double>>& X, vector<double>& y) {
    ifstream archivo(ruta_archivo);
    if (!archivo.is_open()) {
        return false;
    }

    string linea;
    // Leer y descartar la primera línea (encabezado del CSV)
    if (!getline(archivo, linea)) {
        return false;
    }

    while (getline(archivo, linea)) {
        if (linea.empty() || linea == "\r") continue;
        stringstream ss(linea);
        string celda;
        vector<double> fila_valores;

        while (getline(ss, celda, ',')) {
            try {
                fila_valores.push_back(stod(celda));
            } catch (...) {
                // Ignorar celdas vacías o no numéricas
            }
        }

        if (fila_valores.size() >= 2) {
            // Las primeras N-1 columnas son variables de entrada (X) y la última es la salida (y)
            vector<double> caracteristicas(fila_valores.begin(), fila_valores.end() - 1);
            double salida = fila_valores.back();
            X.push_back(caracteristicas);
            y.push_back(salida);
        }
    }

    archivo.close();
    return !X.empty();
}

// Divide el dataset en 80% entrenamiento y 20% prueba de forma intercalada
// (1 de cada 5 muestras va a prueba) para cubrir todo el rango de datos uniformemente
void dividir_entrenamiento_prueba(
    const vector<vector<double>>& X,
    const vector<double>& y,
    vector<vector<double>>& X_train,
    vector<double>& y_train,
    vector<vector<double>>& X_test,
    vector<double>& y_test
) {
    X_train.clear();
    y_train.clear();
    X_test.clear();
    y_test.clear();

    for (size_t i = 0; i < X.size(); i++) {
        if ((i + 1) % 5 == 0) {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        } else {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        }
    }
}

void imprimir_pesos(const vector<double>& pesos) {
    cout << "[ ";
    for (size_t i = 0; i < pesos.size(); i++) {
        cout << fixed << setprecision(6) << pesos[i];
        if (i + 1 < pesos.size()) cout << ", ";
    }
    cout << " ]";
}


// FUNCIÓN PRINCIPAL (main)

int main() {
    cout << "======================================================================" << endl;
    cout << "  PRACTICA 2: PROGRAMACION ORIENTADA A OBJETOS - REGRESION LINEAL" << endl;
    cout << "======================================================================" << endl;

   

    
    cout << "\n--- 1. REGRESION LINEAL SIMPLE (ice_cream_selling_data.csv) ---" << endl;
    vector<vector<double>> X_helados, X_train_h, X_test_h;
    vector<double> y_helados, y_train_h, y_test_h;

    if (!cargar_csv("ice_cream_selling_data.csv", X_helados, y_helados)) {
        cargar_csv("Ice_cream_selling_data.csv", X_helados, y_helados);
    }

    cout << "Total de registros cargados: " << X_helados.size() << endl;
    cout << "Variables de entrada (columnas en X): " << X_helados[0].size() << " (Temperature C)" << endl;

    LinearRegression* slr = new LinearRegression(1);
    // Escalar los datos de entrada antes de dividir y entrenar
    slr->data_scaling(X_helados);
    dividir_entrenamiento_prueba(X_helados, y_helados, X_train_h, y_train_h, X_test_h, y_test_h);

    cout << "Muestras de entrenamiento (80%): " << X_train_h.size()
         << " | Muestras de prueba (20%): " << X_test_h.size() << endl;

    slr->fit(X_train_h, y_train_h, 0.05, 2000);

    cout << "weights: ";
    imprimir_pesos(slr->getWeigths());
    cout << endl;
    cout << "bias: " << fixed << setprecision(6) << slr->getBias() << endl;

    vector<double> y_hat_h = slr->predict(X_test_h);
    cout << "\nPredicciones en el conjunto de prueba (X_test):" << endl;
    for (size_t i = 0; i < y_hat_h.size(); i++) {
        cout << "  Muestra " << setw(2) << (i + 1)
             << " | y_real: " << setw(10) << setprecision(4) << y_test_h[i]
             << " | y_hat: " << setw(10) << setprecision(4) << y_hat_h[i] << endl;
    }
    cout << "score (MSE en prueba): " << fixed << setprecision(6) << slr->score(y_test_h, y_hat_h) << endl;
    cout << "score (MSE en entrenamiento): " << fixed << setprecision(6) << slr->score(y_train_h, slr->predict(X_train_h)) << endl;

    delete slr;

    

    
    cout << "\n--- 2. REGRESION LINEAL MULTIPLE (student_exam_scores.csv) ---" << endl;
    vector<vector<double>> X_estudiantes, X_train_e, X_test_e;
    vector<double> y_estudiantes, y_train_e, y_test_e;

    cargar_csv("student_exam_scores.csv", X_estudiantes, y_estudiantes);

    cout << "Total de registros cargados: " << X_estudiantes.size() << endl;
    cout << "Variables de entrada (columnas en X): " << X_estudiantes[0].size()
         << " (hours_studied, sleep_hours, attendance_percent, previous_scores)" << endl;

    LinearRegression* mlr = new LinearRegression(4);
    // Escalar los datos de entrada
    mlr->data_scaling(X_estudiantes);
    dividir_entrenamiento_prueba(X_estudiantes, y_estudiantes, X_train_e, y_train_e, X_test_e, y_test_e);

    cout << "Muestras de entrenamiento (80%): " << X_train_e.size()
         << " | Muestras de prueba (20%): " << X_test_e.size() << endl;

    mlr->fit(X_train_e, y_train_e, 0.05, 2000);

    cout << "weights: ";
    imprimir_pesos(mlr->getWeigths());
    cout << endl;
    cout << "bias: " << fixed << setprecision(6) << mlr->getBias() << endl;

    vector<double> y_hat_e = mlr->predict(X_test_e);
    cout << "\nPrimeras 10 predicciones en el conjunto de prueba (X_test):" << endl;
    for (size_t i = 0; i < 10 && i < y_hat_e.size(); i++) {
        cout << "  Muestra " << setw(2) << (i + 1)
             << " | y_real: " << setw(8) << setprecision(4) << y_test_e[i]
             << " | y_hat: " << setw(8) << setprecision(4) << y_hat_e[i] << endl;
    }
    cout << "score (MSE en prueba): " << fixed << setprecision(6) << mlr->score(y_test_e, y_hat_e) << endl;
    cout << "Coeficiente R^2 en prueba: " << fixed << setprecision(6) << mlr->calcular_r2(y_test_e, y_hat_e) << endl;
    cout << "score (MSE en entrenamiento): " << fixed << setprecision(6) << mlr->score(y_train_e, mlr->predict(X_train_e)) << endl;
    cout << "Coeficiente R^2 en entrenamiento: " << fixed << setprecision(6) << mlr->calcular_r2(y_train_e, mlr->predict(X_train_e)) << endl;

    

    
    cout << "\n======================================================================" << endl;
    cout << "  3.2 ANALISIS DE MEMORIA (GLOBAL, LOCAL, DINAMICA Y VECTOR DINAMICO)" << endl;
    cout << "======================================================================" << endl;

    // Objeto en memoria local (Stack / Pila)
    LinearRegression modelo_local(4);

    // Objeto en memoria dinámica (Heap / Montículo)
    LinearRegression* modelo_dinamico = new LinearRegression(4);

    cout << "\n1. Direcciones de memoria de los objetos LinearRegression:" << endl;
    cout << "   - Objeto en Memoria Global (.bss/.data): " << &modelo_global << endl;
    cout << "   - Objeto en Memoria Local (Stack/Pila) : " << &modelo_local << endl;
    cout << "   - Objeto en Memoria Dinamica (Heap)    : " << modelo_dinamico
         << " (Puntero alojado en Stack en: " << &modelo_dinamico << ")" << endl;

    cout << "\n2. Tamano (sizeof) de cada objeto y sus componentes internos:" << endl;
    cout << "   - sizeof(modelo_global)      : " << sizeof(modelo_global) << " bytes" << endl;
    cout << "   - sizeof(modelo_local)       : " << sizeof(modelo_local) << " bytes" << endl;
    cout << "   - sizeof(*modelo_dinamico)   : " << sizeof(*modelo_dinamico) << " bytes" << endl;
    cout << "   - sizeof(modelo_dinamico ptr): " << sizeof(modelo_dinamico) << " bytes (tamano del puntero en 64 bits)" << endl;
    cout << "   - Desglose interno de LinearRegression:" << endl;
    cout << "       * sizeof(vector<double> weights): " << sizeof(vector<double>) << " bytes (3 punteros internos de 8 bytes)" << endl;
    cout << "       * sizeof(double bias)           : " << sizeof(double) << " bytes (punto flotante IEEE 754)" << endl;
    cout << "       * Total sizeof(LinearRegression): " << sizeof(LinearRegression) << " bytes" << endl;

    // Crear un vector de LinearRegression en memoria dinámica
    int cantidad_modelos = 5;
    vector<LinearRegression>* vector_dinamico = new vector<LinearRegression>(cantidad_modelos, LinearRegression(4));

    cout << "\n3. Vector de LinearRegression en Memoria Dinamica (N = " << cantidad_modelos << " modelos):" << endl;
    cout << "   - Direccion de la estructura controladora del vector (en Heap): " << vector_dinamico << endl;
    cout << "   - Direccion del arreglo contiguo de elementos (buffer en Heap): " << vector_dinamico->data() << endl;
    for (int i = 0; i < cantidad_modelos; i++) {
        cout << "       * Direccion de (*vector_dinamico)[" << i << "]: " << &((*vector_dinamico)[i]) << endl;
    }
    size_t tamano_estructura_vector = sizeof(*vector_dinamico);
    size_t tamano_elementos_vector = vector_dinamico->size() * sizeof(LinearRegression);
    size_t tamano_total_vector = tamano_estructura_vector + tamano_elementos_vector;

    cout << "   - Tamano de la cabecera std::vector (sizeof(*vector_dinamico)) : " << tamano_estructura_vector << " bytes" << endl;
    cout << "   - Tamano de los " << cantidad_modelos << " objetos almacenados (" << cantidad_modelos << " * " << sizeof(LinearRegression) << " bytes)      : " << tamano_elementos_vector << " bytes" << endl;
    cout << "   - Tamano total en Heap (cabecera + bloque de objetos)          : " << tamano_total_vector << " bytes" << endl;

    // Liberar la memoria dinámica reservada con new
    delete vector_dinamico;
    delete modelo_dinamico;
    delete mlr;

    cout << "\nMemoria dinamica liberada correctamente con delete." << endl;
    return 0;
}
