CC = nvcc
OBJECTS = main_matrix_multiplication.o matrix_multiplication.o matrix_multiplication_tiled.o matrix_multiplication_coarsed.o
BIN = main_matrix_mul.out

all:$(BIN)

$(BIN):$(OBJECTS)
	$(CC) -std=c++11 -o $(BIN) $(OBJECTS)

%.o:%.cu
	$(CC) -std=c++11 -c -o $@ $^

clean:
	rm -rf $(BIN) $(OBJECTS)