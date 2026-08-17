CC = nvcc
OBJECTS = main_matrix_multiplication.o matrix_multiplication.o matrix_multiplication_tiled.o matrix_multiplication_coarsed.o
BIN = main_matrix_mul.out
FLAGS = -std=c++11

all:$(BIN)

$(BIN):$(OBJECTS)
	$(CC) $(FALGS) -o $(BIN) $(OBJECTS)

%.o:%.cu
	$(CC) $(FALGS) -c -o $@ $^

clean:
	rm -rf $(BIN) $(OBJECTS)