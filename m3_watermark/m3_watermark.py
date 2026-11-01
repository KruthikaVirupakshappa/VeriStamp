"""VeriStamp M3: stamp a verdict into a photo and read it back.

    from m3_watermark import load_m3, embed, extract

    m3 = load_m3('best.pt')
    stamped = embed(m3, photo, verdict='AI', p_ai=0.91)
    result = extract(m3, stamped)

Photos are PIL images of any size. The model works at 128 by 128 pixels, so embed computes the stamp
on a 128 pixel copy, enlarges only the stamp and adds it to the full-size photo. extract shrinks the
photo to 128 pixels before reading it. The photo itself keeps its original size and detail.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = 128
MESSAGE_BITS = 32
VERDICT_CODES = {'Real': 0, 'AI': 1, 'Uncertain': 2}
CODE_VERDICTS = {code: name for name, code in VERDICT_CODES.items()}


# Payload: verdict 2 bits, P(AI) bucket 4, detector ID 4, image tag 14, CRC-8 checksum 8.

def _int_to_bits(value, width):
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _bits_to_int(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def _crc8(bits):
    register = 0
    for bit in bits:
        feedback = ((register >> 7) & 1) ^ int(bit)
        register = (register << 1) & 255
        if feedback:
            register ^= 0x07
    return _int_to_bits(register, 8)


def encode_payload(verdict, p_ai, detector_id=3, image_tag=0):
    """Return the 32 bits for a verdict ('Real', 'AI' or 'Uncertain') and a P(AI) between 0 and 1."""
    fields = (_int_to_bits(VERDICT_CODES[verdict], 2)
              + _int_to_bits(round(max(0.0, min(1.0, p_ai)) * 15), 4)
              + _int_to_bits(detector_id & 0xF, 4)
              + _int_to_bits(image_tag & 0x3FFF, 14))
    return fields + _crc8(fields)


def decode_payload(bits):
    """Turn 32 bits back into fields. valid is False when the checksum does not match."""
    bits = [int(b) for b in bits]
    code = _bits_to_int(bits[:2])
    valid = code in CODE_VERDICTS and _crc8(bits[:24]) == bits[24:]
    if not valid:
        return {'valid': False, 'verdict': None, 'p_ai': None, 'detector_id': None, 'image_tag': None}
    return {'valid': True, 'verdict': CODE_VERDICTS[code], 'p_ai': _bits_to_int(bits[2:6]) / 15,
            'detector_id': _bits_to_int(bits[6:10]), 'image_tag': _bits_to_int(bits[10:24])}


# Networks, identical to the training notebooks.

class ConvBNRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                                    nn.BatchNorm2d(channels_out), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self, channels=64, blocks=7, message_bits=32):
        super().__init__()
        layers = [ConvBNRelu(3, channels)] + [ConvBNRelu(channels, channels) for _ in range(blocks - 1)]
        layers += [ConvBNRelu(channels, message_bits), nn.AdaptiveAvgPool2d(1)]
        self.layers = nn.Sequential(*layers)
        self.linear = nn.Linear(message_bits, message_bits)

    def forward(self, image):
        return self.linear(self.layers(image).flatten(1))


class UNetEncoder(nn.Module):
    def __init__(self, channels=64, message_bits=32):
        super().__init__()
        c, m = channels, message_bits
        self.inc = nn.Sequential(ConvBNRelu(3 + m, c), ConvBNRelu(c, c))
        self.down1 = nn.Sequential(ConvBNRelu(c, 2 * c, stride=2), ConvBNRelu(2 * c, 2 * c))
        self.down2 = nn.Sequential(ConvBNRelu(2 * c, 4 * c, stride=2), ConvBNRelu(4 * c, 4 * c))
        self.bottleneck = ConvBNRelu(4 * c + m, 4 * c)
        self.up2 = nn.Sequential(ConvBNRelu(4 * c + 2 * c, 2 * c), ConvBNRelu(2 * c, 2 * c))
        self.up1 = nn.Sequential(ConvBNRelu(2 * c + c, c), ConvBNRelu(c, c))
        self.after_concat_layer = ConvBNRelu(c + 3 + m, c)
        self.final_layer = nn.Conv2d(c, 3, kernel_size=1)

    @staticmethod
    def _expand(message, like):
        return message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, like.shape[-2], like.shape[-1])

    def forward(self, image, message):
        x0 = self.inc(torch.cat([image, self._expand(message, image)], dim=1))
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x2 = self.bottleneck(torch.cat([x2, self._expand(message, x2)], dim=1))
        u2 = self.up2(torch.cat([F.interpolate(x2, size=x1.shape[-2:], mode='nearest'), x1], dim=1))
        u1 = self.up1(torch.cat([F.interpolate(u2, size=x0.shape[-2:], mode='nearest'), x0], dim=1))
        return self.final_layer(self.after_concat_layer(torch.cat([self._expand(message, image), u1, image], dim=1)))


class M3:
    def __init__(self, encoder, decoder, device, epoch):
        self.encoder, self.decoder, self.device, self.epoch = encoder, decoder, device, epoch


def load_m3(path, device=None):
    """Load the final model from best.pt. Uses the GPU when one is available."""
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(path, map_location=device, weights_only=False)
    encoder = UNetEncoder(message_bits=MESSAGE_BITS).to(device)
    decoder = Decoder(message_bits=MESSAGE_BITS).to(device)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()
    return M3(encoder, decoder, device, state.get('epoch'))


# Stamping and reading.

def _to_tensor(image, device):
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _to_image(tensor):
    array = (tensor[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def _shrink(x):
    return F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False, antialias=True)


@torch.no_grad()
def embed(m3, image, verdict, p_ai, detector_id=3, image_tag=0):
    """Return a stamped copy of a PIL image, the same size as the input."""
    message = torch.tensor([encode_payload(verdict, p_ai, detector_id, image_tag)],
                           dtype=torch.float32, device=m3.device)
    original = _to_tensor(image, m3.device)
    small = _shrink(original)
    stamped_small = ((m3.encoder(small * 2 - 1, message) + 1) / 2).clamp(0, 1)
    residual = stamped_small - small
    if original.shape[-2:] != residual.shape[-2:]:
        residual = F.interpolate(residual, size=original.shape[-2:], mode='bilinear', align_corners=False)
    return _to_image(original + residual)


@torch.no_grad()
def read_bits(m3, image):
    """Return the 32 bits the decoder reads from a PIL image."""
    small = _shrink(_to_tensor(image, m3.device))
    return (m3.decoder(small * 2 - 1)[0] > 0.5).long().tolist()


def extract(m3, image):
    """Read the stamp from a PIL image. Check result['valid'] before trusting the verdict."""
    bits = read_bits(m3, image)
    result = decode_payload(bits)
    result['bits'] = bits
    return result
