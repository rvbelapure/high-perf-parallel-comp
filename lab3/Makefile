CC = nvcc

default:
	@echo "Usage : "
	@echo "> make saxpy"
	@echo "OR"
	@echo "> make transpose"

saxpy: cuda_utils.h saxpy.cu timer.c timer.h
	nvcc saxpy.cu -o saxpy

transpose: cuda_utils.h timer.c timer.h transpose.cu
	nvcc transpose.cu -o transpose

clean:
	rm transpose saxpy
