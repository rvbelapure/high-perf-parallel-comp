#include <hpcdefs.hpp>
#include <image.hpp>

void convert_to_floating_point_optimized(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {
	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] =
					double(fixed_point_images[(image_number * image_height + image_row) * image_width + image_column]) / 255.0;
			}
		}
	}
}

void matrix_vector_multiplication_optimized(double *CSE6230_RESTRICT output_vector, const double *CSE6230_RESTRICT matrix, const double *CSE6230_RESTRICT input_vector, size_t matrix_width, size_t matrix_height) {
	for (size_t i = 0; i < matrix_height; i++) {
		double accumulated_sum = 0.0;
		for (size_t j = 0; j < matrix_width; j++) {
			accumulated_sum += matrix[i * matrix_width + j] * input_vector[j];
		}
		output_vector[i] = accumulated_sum;
	}
}
