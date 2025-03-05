#include <iostream>

#include "rtpdsp.h"

rtp_packet mk_rtp_packet(uint32_t ssrc, uint16_t seqnum) {
    // Clear the packet structure
    memset(pkt, 0, sizeof(rtp_packet));

    // Set the header fields
    pkt->header[0] = 0x80; // Version 2, no padding, no extensions, CC=0
    pkt->header[1] = (pt & 0x7F); // Payload type (7 bits)
    pkt->header[2] = (seqnum >> 8) & 0xFF; // Sequence number high byte
    pkt->header[3] = seqnum & 0xFF; // Sequence number low byte

    uint32_t timestamp_network = htonl(timestamp); // Convert timestamp to network byte order
    memcpy(&pkt->header[4], &timestamp_network, sizeof(uint32_t)); // Copy timestamp


    uint32_t ssrc_network = htonl(ssrc); // Convert SSRC to network byte order
    memcpy(&pkt->header[8], &ssrc_network, sizeof(uint32_t)); // Copy SSRC

    // Copy the payload
    strncpy(pkt->payload, payload, min(sizeof(pkt->payload), strlen(payload)));
    //printf("Payload: %s len=%ld\n", pkt->payload, strlen(pkt->payload));

    return RTP_HEADER_SIZE + strlen(payload);
}


int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}