#include <hpcdefs.hpp>
#include <image.hpp>
#include <x86intrin.h>
#include <string.h>

#define DEBUG_GRAYSCALE 0
#define BY_PARTS 0

/* prints 128bit var by treating it as vector of cast_to_bits bit numbers */
void print128_num(const char * message, __m128i var, int cast_to_bits = 8)
{
	printf("%s :", message);
	if(cast_to_bits == 8)
	{
		uint8_t *val = (uint8_t *) &var;
		for(size_t i = 0 ; i < 128 / cast_to_bits ; i++)
			printf("%i ",val[i]);
		printf("\n");
	}
	else if(cast_to_bits == 16)
	{
		uint16_t *val = (uint16_t *) &var;
		for(size_t i = 0 ; i < 128 / cast_to_bits ; i++)
			printf("%i ",val[i]);
		printf("\n");
	}
	else if(cast_to_bits == 32)
	{
		uint32_t *val = (uint32_t *) &var;
		for(size_t i = 0 ; i < 128 / cast_to_bits ; i++)
			printf("%i ",val[i]);
		printf("\n");
	}
	else if(cast_to_bits == 64)
	{
		uint64_t *val = (uint64_t *) &var;
		for(size_t i = 0 ; i < 128 / cast_to_bits ; i++)
			printf("%i ",val[i]);
		printf("\n");
	}
}

void convert_rgb_to_grayscale_optimized(const uint8_t *CSE6230_RESTRICT rgb_image, uint8_t *CSE6230_RESTRICT grayscale_image, size_t width, size_t height) {

	size_t total_pixels = width * height * 3;
	size_t cols = total_pixels / 48;

	__m128i chunk[cols][3];
	__m128i red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
	__m128i red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
	__m128i red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
	__m128i green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
	__m128i green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
	__m128i blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
	__m128i blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

	__m128i red8[cols], green8[cols], blue8[cols];
	__m128i red[cols][2], green[cols][2], blue[cols][2], luma[cols][2];
	__m128i zero = _mm_setzero_si128();
	__m128i fiftyfour = _mm_set1_epi16(54);
	__m128i oneeigthythree = _mm_set1_epi16(183);
	__m128i nineteen = _mm_set1_epi16(19);
	__m128i luma_shifted[cols][2], luma8[cols];

	__m128i move_right = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
	__m128i move_left = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1);

	uint8_t *output = grayscale_image;
	for(size_t i = 0 ; i < cols ; i++)
	{
		/* 1. Load all uint8's in 128bit vectors. As our total image size is divisible by 48, we load in chunks of 3 */
		chunk[i][0] = _mm_loadu_si128((const __m128i *) rgb_image);
		chunk[i][1] = _mm_loadu_si128((const __m128i *) (rgb_image + 16));
		chunk[i][2] = _mm_loadu_si128((const __m128i *) (rgb_image + 32));
		rgb_image += 48;

		/* 2. De-interleave the rgb into sets of 128bit vectors containing uint8's */
		red8[i] = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[i][0], red_indeces_0),
					_mm_shuffle_epi8(chunk[i][1], red_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[i][2], red_indeces_2)
				);
		green8[i] = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[i][0], green_indeces_0),
					_mm_shuffle_epi8(chunk[i][1], green_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[i][2], green_indeces_2)
				);
		blue8[i] = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[i][0], blue_indeces_0),
					_mm_shuffle_epi8(chunk[i][1], blue_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[i][2], blue_indeces_2)
				);

		/* 3. Now we have de-interleaved rgb in 8bit format. now we have to convert them in 16 bit format
		 * Thus, each red, green and blue will split into two 128bit vectors each */
		red[i][0] = _mm_unpacklo_epi8(red8[i], zero);
		green[i][0] = _mm_unpacklo_epi8(green8[i], zero);
		blue[i][0] = _mm_unpacklo_epi8(blue8[i], zero);

		red[i][1] = _mm_unpackhi_epi8(red8[i], zero);
		green[i][1] = _mm_unpackhi_epi8(green8[i], zero);
		blue[i][1] = _mm_unpackhi_epi8(blue8[i], zero);

		/* 4. red, blue and green now have everything in 16bit integer format. We now perform multiply and shift op */
		luma[i][0] = _mm_srli_epi16(
				_mm_add_epi16(
					_mm_add_epi16(
							_mm_mullo_epi16(fiftyfour, red[i][0]),
							_mm_mullo_epi16(oneeigthythree, green[i][0])
						     ),
					_mm_mullo_epi16(nineteen, blue[i][0])
					), 8);

		luma[i][1] = _mm_srli_epi16(
				_mm_add_epi16(
					_mm_add_epi16(
							_mm_mullo_epi16(fiftyfour, red[i][1]),
							_mm_mullo_epi16(oneeigthythree, green[i][1])
						     ),
					_mm_mullo_epi16(nineteen, blue[i][1])
					), 8);

		/* 5. We now shuffle the 16 bit values and zeros to get 8 bit values together */
		luma_shifted[i][0] = _mm_shuffle_epi8(luma[i][0], move_right);
		luma_shifted[i][1] = _mm_shuffle_epi8(luma[i][1], move_left);

		/* 6. OR the two results above to get 8 bit contiguous result */
		luma8[i] = _mm_or_si128(luma_shifted[i][0], luma_shifted[i][1]);

		/* 7. store the result back */
		_mm_storeu_si128((__m128i *) output, luma8[i]);
		output += 16;
	}
/*
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			const uint16_t red   = rgb_image[(i * width + j) * 3 + 0];
			const uint16_t green = rgb_image[(i * width + j) * 3 + 1];
			const uint16_t blue  = rgb_image[(i * width + j) * 3 + 2];
			const uint8_t  luma  = (red * 54 + green * 183 + blue * 19) >> 8;
			grayscale_image[i * width + j] = luma;
		}
	}
*/
}

void integrate_image_optimized(const uint8_t *CSE6230_RESTRICT source_image, uint32_t *CSE6230_RESTRICT integral_image, size_t width, size_t height) {
	for (size_t i = 0; i < height; i++) {
		uint32_t integral = 0;
		for (size_t j = 0; j < width; j++) {
			integral += source_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}
	for (size_t j = 0; j < width; j++) {
		uint32_t integral = 0;
		for (size_t i = 0; i < height; i++) {
			integral += integral_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}
}
