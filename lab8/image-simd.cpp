#include <hpcdefs.hpp>
#include <image.hpp>
#include <x86intrin.h>
#include <string.h>

void convert_rgb_to_grayscale_optimized(const uint8_t *CSE6230_RESTRICT rgb_image, uint8_t *CSE6230_RESTRICT grayscale_image, size_t width, size_t height) {

	size_t total_pixels = width * height * 3;
	size_t cols = total_pixels / 48;

	__m128i chunk[3];
	__m128i red_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0);
	__m128i red_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1);
	__m128i red_indeces_2 = _mm_set_epi8(13, 10, 7, 4, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i green_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1);
	__m128i green_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, 15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1);
	__m128i green_indeces_2 = _mm_set_epi8(14, 11, 8, 5, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
	__m128i blue_indeces_0 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 14, 11, 8, 5, 2);
	__m128i blue_indeces_1 = _mm_set_epi8(-1, -1, -1, -1, -1, -1, 13, 10, 7, 4, 1, -1, -1, -1, -1, -1);
	__m128i blue_indeces_2 = _mm_set_epi8(15, 12, 9, 6, 3, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);

	__m128i red8, green8, blue8;
	__m128i red[2], green[2], blue[2], luma[2];
	__m128i zero = _mm_setzero_si128();
	__m128i fiftyfour = _mm_set1_epi16(54);
	__m128i oneeigthythree = _mm_set1_epi16(183);
	__m128i nineteen = _mm_set1_epi16(19);
	__m128i luma_shifted[2], luma8;

	__m128i move_right = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0);
	__m128i move_left = _mm_set_epi8(14, 12, 10, 8, 6, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1, -1);

	uint8_t *output = grayscale_image;
	for(size_t i = 0 ; i < cols ; i++)
	{
		/* 1. Load all uint8's in 128bit vectors. As our total image size is divisible by 48, we load in chunks of 3 */
		chunk[0] = _mm_loadu_si128((const __m128i *) rgb_image);
		chunk[1] = _mm_loadu_si128((const __m128i *) (rgb_image + 16));
		chunk[2] = _mm_loadu_si128((const __m128i *) (rgb_image + 32));
		rgb_image += 48;

		/* 2. De-interleave the rgb into sets of 128bit vectors containing uint8's */
		red8 = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[0], red_indeces_0),
					_mm_shuffle_epi8(chunk[1], red_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[2], red_indeces_2)
				);
		green8 = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[0], green_indeces_0),
					_mm_shuffle_epi8(chunk[1], green_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[2], green_indeces_2)
				);
		blue8 = _mm_or_si128( 
				_mm_or_si128(
					_mm_shuffle_epi8(chunk[0], blue_indeces_0),
					_mm_shuffle_epi8(chunk[1], blue_indeces_1)
					    ),
				_mm_shuffle_epi8(chunk[2], blue_indeces_2)
				);

		/* 3. Now we have de-interleaved rgb in 8bit format. now we have to convert them in 16 bit format
		 * Thus, each red, green and blue will split into two 128bit vectors each */
		red[0] = _mm_unpacklo_epi8(red8, zero);
		green[0] = _mm_unpacklo_epi8(green8, zero);
		blue[0] = _mm_unpacklo_epi8(blue8, zero);

		red[1] = _mm_unpackhi_epi8(red8, zero);
		green[1] = _mm_unpackhi_epi8(green8, zero);
		blue[1] = _mm_unpackhi_epi8(blue8, zero);

		/* 4. red, blue and green now have everything in 16bit integer format. We now perform multiply and shift op */
		luma[0] = _mm_srli_epi16(
				_mm_add_epi16(
					_mm_add_epi16(
							_mm_mullo_epi16(fiftyfour, red[0]),
							_mm_mullo_epi16(oneeigthythree, green[0])
						     ),
					_mm_mullo_epi16(nineteen, blue[0])
					), 8);

		luma[1] = _mm_srli_epi16(
				_mm_add_epi16(
					_mm_add_epi16(
							_mm_mullo_epi16(fiftyfour, red[1]),
							_mm_mullo_epi16(oneeigthythree, green[1])
						     ),
					_mm_mullo_epi16(nineteen, blue[1])
					), 8);

		/* 5. We now shuffle the 16 bit values and zeros to get 8 bit values together */
		luma_shifted[0] = _mm_shuffle_epi8(luma[0], move_right);
		luma_shifted[1] = _mm_shuffle_epi8(luma[1], move_left);

		/* 6. OR the two results above to get 8 bit contiguous result */
		luma8 = _mm_or_si128(luma_shifted[0], luma_shifted[1]);

		/* 7. store the result back */
		_mm_storeu_si128((__m128i *) output, luma8);
		output += 16;
	}

/*	for (size_t i = 0; i < height; i++) {
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

	__m128i chunk1, chunk2;
	__m128i chunk1_16[2], chunk1_32[4];
	__m128i chunk2_16[2], chunk2_32[4];
	__m128i zero = _mm_setzero_si128();

	for(size_t i = 0 ; i < height ; i++)
	{

		__m128i sum = zero;
		for(size_t j = 0 ; (j+16) < width ; j += 16)
		{
			/* load ith and (i-1)th row */
			int curr_start = i*width + j;
			int prev_start = (i-1)*width + j;
			if(prev_start < 0)
			{
				chunk1_32[0] = _mm_setzero_si128();
				chunk1_32[1] = _mm_setzero_si128();
				chunk1_32[2] = _mm_setzero_si128();
				chunk1_32[3] = _mm_setzero_si128();
			}
			else
			{
				chunk1_32[0] = _mm_loadu_si128((const __m128i *) (integral_image + prev_start));
				chunk1_32[1] = _mm_loadu_si128((const __m128i *) (integral_image + prev_start + 4));
				chunk1_32[2] = _mm_loadu_si128((const __m128i *) (integral_image + prev_start + 8));
				chunk1_32[3] = _mm_loadu_si128((const __m128i *) (integral_image + prev_start + 12));
			}

			chunk2 = _mm_loadu_si128((const __m128i *) (source_image + curr_start));

			/* convert the loaded data in 32 bit representation */
			chunk2_16[0] = _mm_unpacklo_epi8(chunk2, zero);
			chunk2_16[1] = _mm_unpackhi_epi8(chunk2, zero);
			chunk2_32[0] = _mm_unpacklo_epi16(chunk2_16[0], zero);
			chunk2_32[1] = _mm_unpackhi_epi16(chunk2_16[0], zero);
			chunk2_32[2] = _mm_unpacklo_epi16(chunk2_16[1], zero);
			chunk2_32[3] = _mm_unpackhi_epi16(chunk2_16[1], zero);

			/* horizontal prefix sum */
			__m128i shifted0 = _mm_slli_si128(chunk2_32[0], 4);
			__m128i shifted1 = _mm_slli_si128(chunk2_32[1], 4);
			__m128i shifted2 = _mm_slli_si128(chunk2_32[2], 4);
			__m128i shifted3 = _mm_slli_si128(chunk2_32[3], 4);
			for(size_t k = 0 ; k < 3 ; k++)
			{
				chunk2_32[0] = _mm_add_epi32(chunk2_32[0], shifted0);
				chunk2_32[1] = _mm_add_epi32(chunk2_32[1], shifted1);
				chunk2_32[2] = _mm_add_epi32(chunk2_32[2], shifted2);
				chunk2_32[3] = _mm_add_epi32(chunk2_32[3], shifted3);
				shifted0 = _mm_slli_si128(shifted0, 4);
				shifted1 = _mm_slli_si128(shifted1, 4);
				shifted2 = _mm_slli_si128(shifted2, 4);
				shifted3 = _mm_slli_si128(shifted3, 4);
			}
			chunk2_32[0] = _mm_add_epi32(chunk2_32[0], sum);
			sum = _mm_set1_epi32(_mm_extract_epi32(chunk2_32[0], 3));
			chunk2_32[1] = _mm_add_epi32(chunk2_32[1], sum);
			sum = _mm_set1_epi32(_mm_extract_epi32(chunk2_32[1], 3));
			chunk2_32[2] = _mm_add_epi32(chunk2_32[2], sum);
			sum = _mm_set1_epi32(_mm_extract_epi32(chunk2_32[2], 3));
			chunk2_32[3] = _mm_add_epi32(chunk2_32[3], sum);
			sum = _mm_set1_epi32(_mm_extract_epi32(chunk2_32[3], 3));

			/* vertical sum */
			chunk2_32[0] = _mm_add_epi32(chunk1_32[0], chunk2_32[0]);
			chunk2_32[1] = _mm_add_epi32(chunk1_32[1], chunk2_32[1]);
			chunk2_32[2] = _mm_add_epi32(chunk1_32[2], chunk2_32[2]);
			chunk2_32[3] = _mm_add_epi32(chunk1_32[3], chunk2_32[3]);

			/* store back in integral image */
			_mm_storeu_si128((__m128i *) (integral_image + curr_start), chunk2_32[0]);
			_mm_storeu_si128((__m128i *) (integral_image + curr_start + 4), chunk2_32[1]);
			_mm_storeu_si128((__m128i *) (integral_image + curr_start + 8), chunk2_32[2]);
			_mm_storeu_si128((__m128i *) (integral_image + curr_start + 12), chunk2_32[3]);
		}
	}

	size_t start_addr = ((width / 16 ) * 16) -1;

	for(size_t j = start_addr ; j < width ; j++)
	{
		uint32_t integral = 0;
		for (size_t i = 0; i < height; i++) {
			integral += source_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}

	for(size_t i = 0 ; i < height ; i++)
	{
		uint32_t integral = integral_image[i*width + start_addr -1];
		for(size_t j = start_addr; j < width ; j++)
		{
			integral += integral_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}

/*
 	
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
*/
}
