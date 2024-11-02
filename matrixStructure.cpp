#include <iostream>
#include <stdexcept>
#include <random>
#include <cmath>

class matrix {
public:
    int rows;
    int columns;
    double** data;

    // constructor
    matrix(int r, int c) : rows(r), columns(c) {
        data = new double*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new double[columns];
            for (int j = 0; j < columns; j++) {
                data[i][j] = 0.0;
            }
        }
    }

    // destructor
    ~matrix() {
        for (int i = 0; i < rows; i++) {
            delete[] data[i];
        }
        delete[] data;
    }

    // copy constructor
    matrix(const matrix& other) : rows(other.rows), columns(other.columns) {
        data = new double*[rows];
        for (int i = 0; i < rows; i++) {
            data[i] = new double[columns];
            for (int j = 0; j < columns; j++) {
                data[i][j] = other.data[i][j];
            }
        }
    }

    // copy assignment operator
    matrix& operator=(const matrix& other) {
        if (this != &other) {
            for (int i = 0; i < rows; i++) {
                delete[] data[i];
            }
            delete[] data;

            rows = other.rows;
            columns = other.columns;
            data = new double*[rows];
            for (int i = 0; i < rows; i++) {
                data[i] = new double[columns];
                for (int j = 0; j < columns; j++) {
                    data[i][j] = other.data[i][j];
                }
            }
        }
        return *this;
    }

    // move constructor
    matrix(matrix&& other) noexcept : rows(other.rows), columns(other.columns), data(other.data) {
        other.data = nullptr;
        other.rows = 0;
        other.columns = 0;
    }

    // move assignment operator
    matrix& operator=(matrix&& other) noexcept {
        if (this != &other) {
            if (data != nullptr) {
                for (int i = 0; i < rows; i++) {
                    delete[] data[i];
                }
                delete[] data;
            }

            rows = other.rows;
            columns = other.columns;
            data = other.data;

            other.data = nullptr;
            other.rows = 0;
            other.columns = 0;
        }
        return *this;
    }

    // subscript operators
    double* operator[](int index) { return data[index]; }
    const double* operator[](int index) const { return data[index]; }
};

// print matrix
void print(const matrix& A) {
    for (int i = 0; i < A.rows; i++) {
        std::cout << "|";
        for (int j = 0; j < A.columns; j++) {
            std::cout << " " << A.data[i][j] << " ";
        }
        std::cout << "|" << std::endl;
    }
}

// matrix multiplication
matrix multiply(const matrix& A, const matrix& B) {
    if (A.columns != B.rows) {
        throw std::invalid_argument("\nMatrix dimensions do not match for multiplication!\n");
    }

    matrix z(A.rows, B.columns);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.columns; j++) {
            for (int k = 0; k < A.columns; k++) {
                z[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return z;
}

// matrix addition
matrix add(const matrix& A, const matrix& B) {
    if (A.columns != B.columns || A.rows != B.rows) {
        throw std::invalid_argument("\nMatrix dimensions do not match for addition!\n");
    }

    matrix z(A.rows, B.columns);
    for (int i = 0; i < A.rows; i++) {
        for (int j = 0; j < B.columns; j++) {
            z[i][j] = A[i][j] + B[i][j];
        }
    }
    return z;
}
