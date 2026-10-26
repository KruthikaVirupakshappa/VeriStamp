"""The 32-bit stamp payload and the "is this stamp valid?" rule (owners: C + D).

Layout (TEAM DECISION; this is the guide's suggested default):

    bits  0-1   verdict        00 Real, 01 AI, 10 Uncertain, 11 reserved (invalid)
    bits  2-5   p_ai bucket    round(P(AI) * 15), so 0..15
    bits  6-9   detector id    0 stub, 1 M1, 2 M2, 3 Fusion
    bits 10-23  image tag      14-bit tag (e.g. from the image id); helps spot copied stamps
    bits 24-31  CRC-8          checksum over bits 0-23 (polynomial 0x07)

A stamp is VALID only if the CRC recomputed from bits 0-23 equals bits 24-31
and the verdict code is not reserved. Every part of the project (pipeline,
RQ2 invalidation rate, RQ3, demo) must use `decode()` so "valid" means one thing.

Validity is NOT authenticity: the stamp is unsigned, so anyone with the encoder
can write any verdict. A valid stamp only says "our detector said X when stamped".
"""
from __future__ import annotations

import zlib

N_BITS = 32
VERDICT_CODES = {"Real": 0, "AI": 1, "Uncertain": 2}
CODE_TO_VERDICT = {v: k for k, v in VERDICT_CODES.items()}
DETECTOR_IDS = {"stub": 0, "m1_resnet": 1, "m2_clip": 2, "fusion": 3}


def int_to_bits(value: int, n: int) -> list[int]:
    if not 0 <= value < 2 ** n:
        raise ValueError(f"{value} does not fit in {n} bits")
    return [(value >> (n - 1 - i)) & 1 for i in range(n)]


def bits_to_int(bits) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def crc8(bits) -> list[int]:
    """Bitwise CRC-8 (x^8 + x^2 + x + 1, init 0) over a list of 0/1 values."""
    c = 0
    for b in bits:
        top = ((c >> 7) & 1) ^ int(b)
        c = (c << 1) & 0xFF
        if top:
            c ^= 0x07
    return int_to_bits(c, 8)


def image_tag_for(image_id: str) -> int:
    return zlib.crc32(image_id.encode()) & 0x3FFF


def encode(verdict: str, p_ai: float, detector: str, image_tag: int = 0) -> list[int]:
    if verdict not in VERDICT_CODES:
        raise ValueError(f"unknown verdict {verdict!r}")
    if detector not in DETECTOR_IDS:
        raise ValueError(f"unknown detector {detector!r}; add it to DETECTOR_IDS")
    bucket = int(round(min(max(p_ai, 0.0), 1.0) * 15))
    data = (int_to_bits(VERDICT_CODES[verdict], 2) + int_to_bits(bucket, 4)
            + int_to_bits(DETECTOR_IDS[detector], 4) + int_to_bits(image_tag & 0x3FFF, 14))
    return data + crc8(data)


def decode(bits) -> dict:
    """bits: 32 values of 0/1. Returns the fields only when the stamp is valid."""
    bits = [int(b) for b in bits]
    if len(bits) != N_BITS:
        raise ValueError(f"expected {N_BITS} bits, got {len(bits)}")
    data, check = bits[:24], bits[24:]
    code = bits_to_int(data[0:2])
    valid = crc8(data) == check and code in CODE_TO_VERDICT
    if not valid:
        return {"valid": False, "verdict": None, "p_ai_bucket": None, "detector_id": None, "image_tag": None}
    return {"valid": True, "verdict": CODE_TO_VERDICT[code], "p_ai_bucket": bits_to_int(data[2:6]),
            "detector_id": bits_to_int(data[6:10]), "image_tag": bits_to_int(data[10:24])}
