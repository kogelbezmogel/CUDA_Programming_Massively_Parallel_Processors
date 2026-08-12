CC = nvcc
OBJECTS = main_matrix_multiplication.o matrix_multiplication.o matrix_multiplication_tiled.o
BIN = main_matrix_mul 

all:$(BIN)

$(BIN):$(OBJECTS)
	$(CC) -o $(BIN) $(OBJECTS)

%.o:%.cu
	$(CC) -c -o $@ $^

clean:
	rm -rf $(BIN) $(OBJECTS)