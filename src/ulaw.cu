/*
 * From Sun Microsystems g711 code (see https://github.com/escrichov/G711/blob/master/g711.c)
 */

#include "ulaw.h"

#include <cstdint>

/* part of Sun Microsystems g711 code (https://github.com/escrichov/G711/blob/master/g711.c) */
int16_t search(int16_t val, int16_t *table, int16_t size)
{
    int16_t i;

    for (i = 0; i < size; i++) {
        if (val <= *table++)
            return (i);
    }
    return (size);
}

uint8_t pcm2ulaw(int16_t pcm_val)	/* 2's complement (16-bit range) */
{
    int16_t mask;
    int16_t seg;
    uint8_t uval;

    /* Get the sign and the magnitude of the value. */
    pcm_val = pcm_val >> 2;
    if (pcm_val < 0) {
        pcm_val = -pcm_val;
        mask = 0x7F;
    } else {
        mask = 0xFF;
    }
    if ( pcm_val > CLIP ) pcm_val = CLIP;		/* clip the magnitude */
    pcm_val += (BIAS >> 2);

    /* Convert the scaled magnitude to segment number. */
    seg = search(pcm_val, seg_uend, 8);

    /*
    * Combine the sign, segment, quantization bits;
    * and complement the code word.
    */
    if (seg >= 8)		/* out of range, return maximum value. */
        return (unsigned char) (0x7F ^ mask);
    else {
        uval = (unsigned char) (seg << 4) | ((pcm_val >> (seg + 1)) & 0xF);
        return (uval ^ mask);
    }

}

__host__ __device__ int16_t ulaw2pcm(uint8_t u_val)
{
    int16_t t;

    /* Complement to obtain normal u-law value. */
    u_val = ~u_val;

    /*
     * Extract and bias the quantization bits. Then
     * shift up by the segment number and subtract out the bias.
     */
    t = ((u_val & QUANT_MASK) << 3) + BIAS;
    t <<= ((unsigned)u_val & SEG_MASK) >> SEG_SHIFT;

    return ((u_val & SIGN_BIT) ? (BIAS - t) : (t - BIAS));
}