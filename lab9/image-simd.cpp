#include <hpcdefs.hpp>
#include <image.hpp>

static const __m128i shuff1 = _mm_set_epi8(-1, -1, -1, 3, -1, -1, -1, 2, -1, -1, -1, 1, -1, -1, -1, 0);
static const __m128i shuff2 = _mm_set_epi8(-1, -1, -1, 7, -1, -1, -1, 6, -1, -1, -1, 5, -1, -1, -1, 4);
static const __m128i shuff3 = _mm_set_epi8(-1, -1, -1, 11, -1, -1, -1, 10, -1, -1, -1, 9, -1, -1, -1, 8);
static const __m128i shuff4 = _mm_set_epi8(-1, -1, -1, 15, -1, -1, -1, 14, -1, -1, -1, 13, -1, -1, -1, 12);
static const __m128d factor = _mm_set1_pd( (double) ((double) 1.0 / (double) 255.0));

void convert_to_floating_point_optimized(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {

	size_t total_Count = image_count * image_height * image_width;
	for(size_t offset = 0 ; offset < total_Count ; offset += 16)
	{
		/* Load 128 bits into vector - 16 eight bit unsigned ints*/
		__m128i chunk = _mm_load_si128((const __m128i *) (fixed_point_images + offset));

		/* shuffle to convert it to 4 chunks of four 32 bit unsigned ints */
		__m128i part1_32 = _mm_shuffle_epi8(chunk, shuff1);
		__m128i part2_32 = _mm_shuffle_epi8(chunk, shuff2);
		__m128i part3_32 = _mm_shuffle_epi8(chunk, shuff3);
		__m128i part4_32 = _mm_shuffle_epi8(chunk, shuff4);

		
		/* convert 32 bit uints to 64 bit doubles. Then divide them by 255. (Actually, multiply by 1/255) 
		 * The convert and divide operations are interleaved to get good L1-hit rate. 
		 * This increases performance by 0.1x to 0.2x over what we get without interleaving. */
		__m128d part1_d_lo = _mm_cvtepi32_pd(part1_32);
		__m128d part1_d_hi = _mm_cvtepi32_pd( _mm_srli_si128(part1_32,8) );
		part1_d_lo = _mm_mul_pd(part1_d_lo, factor);
		part1_d_hi = _mm_mul_pd(part1_d_hi, factor);

		__m128d part2_d_lo = _mm_cvtepi32_pd(part2_32);
		__m128d part2_d_hi = _mm_cvtepi32_pd( _mm_srli_si128(part2_32,8) );
		part2_d_lo = _mm_mul_pd(part2_d_lo, factor);
		part2_d_hi = _mm_mul_pd(part2_d_hi, factor);

		__m128d part3_d_lo = _mm_cvtepi32_pd(part3_32);
		__m128d part3_d_hi = _mm_cvtepi32_pd( _mm_srli_si128(part3_32,8) );
		part3_d_lo = _mm_mul_pd(part3_d_lo, factor);
		part3_d_hi = _mm_mul_pd(part3_d_hi, factor);

		__m128d part4_d_lo = _mm_cvtepi32_pd(part4_32);
		__m128d part4_d_hi = _mm_cvtepi32_pd( _mm_srli_si128(part4_32,8) );
		part4_d_lo = _mm_mul_pd(part4_d_lo, factor);
		part4_d_hi = _mm_mul_pd(part4_d_hi, factor);

		/* store everything back */
		_mm_store_pd((double *) (floating_point_images + offset), part1_d_lo);
		_mm_store_pd((double *) (floating_point_images + offset + 2), part1_d_hi);
		_mm_store_pd((double *) (floating_point_images + offset + 4), part2_d_lo);
		_mm_store_pd((double *) (floating_point_images + offset + 6), part2_d_hi);
		_mm_store_pd((double *) (floating_point_images + offset + 8), part3_d_lo);
		_mm_store_pd((double *) (floating_point_images + offset + 10), part3_d_hi);
		_mm_store_pd((double *) (floating_point_images + offset + 12), part4_d_lo);
		_mm_store_pd((double *) (floating_point_images + offset + 14), part4_d_hi);
	}

/*	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] =
					double(fixed_point_images[(image_number * image_height + image_row) * image_width + image_column]) / 255.0;
			}
		}
	}
*/
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
