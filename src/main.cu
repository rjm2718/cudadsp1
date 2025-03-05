#include <iostream>
#include <cmath>
#include <arpa/inet.h>

#include "rtpdsp.h"

uint8_t pcm2ulaw(int16_t pcm) {
    uint8_t ulawByte;
    const int cBias = 0x84;
    const int cClip = 32635;

    // Clamp the PCM value
    if (pcm < -cClip) pcm = -cClip;
    if (pcm > cClip) pcm = cClip;

    // Get the sign and the magnitude of the PCM value
    int sign = (pcm < 0) ? 0x80 : 0;
    if (sign) pcm = -pcm;

    // Add the bias, then clip
    pcm += cBias;
    if (pcm > 32767) pcm = 32767;

    // Convert the adjusted magnitude to µ-law
    int exponent = 7;
    int mantissa = 0;
    for (int value = pcm >> 7; value > 0; value >>= 1) {
        if (--exponent < 0) break;
    }
    mantissa = (pcm >> (exponent + 3)) & 0x0F;
    ulawByte = ~(sign | (exponent << 4) | mantissa);

    return ulawByte;
}




void add_sine_wave(int16_t *buffer, int length, double amplitude, double frequency, double sample_rate) {
    const double two_pi = 2.0 * M_PI;
    for (int i = 0; i < length; i++) {
        double t = static_cast<double>(i) / sample_rate; // Time for current sample
        buffer[i] += static_cast<int16_t>(amplitude * sin(two_pi * frequency * t));
    }
}

void mk_rtp_packet(rtp_packet *pkt, uint32_t ssrc) {

    // Set the header fields
    pkt->header[0] = 0x80; // Version 2, no padding, no extensions, CC=0
    pkt->header[1] = 0; // payload type ulaw

    // increment sequence number
    uint16_t *seqnum = (uint16_t*)&pkt->header[2];
    *seqnum++;

    uint32_t ssrc_network = htonl(ssrc);
    memcpy(&pkt->header[8], &ssrc_network, sizeof(uint32_t)); // Copy SSRC

    // simple sine wave composite
    int16_t buffer[160];
    memset(buffer, 0, sizeof(buffer));
    add_sine_wave(buffer, 160, 6000, 440, 8000);

    double freq = 1000 + ssrc % 2000;
    add_sine_wave(buffer, 160, 18000, freq, 8000);

    // convert to ulaw
    for (int i = 0; i < 160; i++) {
        pkt->payload[i] = pcm2ulaw(buffer[i]);
    }
}

int main() {

    int Pc = 100;

    // large buffer to write packets to
    void* pktbuf_h = malloc(sizeof(rtp_packet) * Pc);
    void* pbuf = pktbuf_h;
    for (int i = 0; i < Pc; i++, pbuf += sizeof(rtp_packet)) {
        uint32_t ssrc = i;
        mk_rtp_packet((rtp_packet*)pbuf, ssrc);
    }

    return 0;
}