# VeriStamp M3: hiding the verdict inside the photo

VeriStamp detects whether a photo is real or AI generated and then hides that verdict inside the photo itself. The detectors, M1, M2 and a fusion model, make the decision. M3, in this folder, hides the decision as an invisible 32-bit message in the pixels and reads it back later. Because the message is in the pixels and not in the file's metadata, it stays with the photo when apps strip the metadata.

The 32 bits hold the verdict, the detector's confidence, a detector ID, a short image tag and a checksum. The checksum tells us whether the message was read without errors.

M3 is based on HiDDeN, a published watermarking model, using the open-source PyTorch implementation by ando-khachatryan.

## Notebooks

Read them in this order. Each one starts with an explanation of how and why it works.

1. [HiDDeN with a convolutional encoder](1_HiDDeN_conv_encoder.ipynb) trains HiDDeN with its original encoder on clean photos.
2. [HiDDeN with a U-Net encoder](2_HiDDeN_unet_encoder.ipynb) replaces the encoder with a U-Net and changes nothing else.
3. [Training for JPEG and resizing](3_final_model_jpeg_resize.ipynb) continues training the U-Net on compressed and resized photos. This produces the final model.
4. [Using the final model](4_using_the_final_model.ipynb) shows how to stamp and read photos at their full size, and tests it on 500 photos.

## Using the final model

`m3_watermark.py` has everything needed to use M3 from other code, such as the RQ3 experiments and the demo:

```python
from m3_watermark import load_m3, embed, extract

m3 = load_m3('best.pt')
stamped = embed(m3, photo, verdict='AI', p_ai=0.91)
result = extract(m3, stamped)
```

`photo` is a PIL image of any size. `result` holds the verdict, the confidence and `valid`, which is False when the message could not be read correctly. Only trust the verdict when `valid` is True.

The model itself works on 128 by 128 pixel images. For larger photos, `embed` works out the stamp on a small copy, enlarges only the stamp and adds it to the full photo, so the photo keeps all its detail. `extract` shrinks the photo back to 128 pixels before reading it.

## Download the model and results

The trained model and the full results are too large for GitHub, so they are on Google Drive: [VeriStamp M3 results and model](https://drive.google.com/file/d/1oB5NEaWLoLlbarJQnbLhmrG4znF0ZaEi/view?usp=drive_link).

The download contains one results ZIP for each of the first three notebooks, with the training logs, test results, figures and model files. The final model is `best.pt` inside `results_3_final_model.zip`. Extract it and pass its path to `load_m3`.

## Results

All tests use 500 COCO photos that were never used for training or for choosing the model. Bit accuracy is the share of the 32 bits read back correctly, where 50% means guessing. A valid verdict means all bits passed the checksum and the verdict was right. PSNR measures how close the stamped photo is to the original; higher means the stamp is harder to see.

### Final model on full-size photos

These come from notebook 4 and show how the model behaves on real photos.

| Edit | Bit accuracy | Valid verdict |
|---|---:|---:|
| Saved as PNG | 99.85% | 96.0% |
| JPEG quality 90 | 99.83% | 95.2% |
| JPEG quality 70 | 99.58% | 90.0% |
| JPEG quality 50 | 98.75% | 72.4% |
| Resized to 224 pixels | 99.83% | 95.4% |

PSNR is 31.50 dB, and none of the 500 photos without a stamp was wrongly read as stamped.

### How the encoders compared

Both encoders were trained on clean photos with the same data and settings, and tested after saving as PNG.

| | Original encoder | U-Net encoder |
|---|---:|---:|
| Bit accuracy | 92.97% | 99.66% |
| All 32 bits correct | 13.2% | 94.8% |
| PSNR | 35.47 dB | 32.65 dB |

The U-Net read the message far more reliably, at the cost of a slightly stronger stamp, so we kept it.

### Training for JPEG and resizing

These tests from notebook 3 use 128 pixel images, which is a harder case than full-size photos.

| Edit | Before this training | After this training |
|---|---:|---:|
| Saved as PNG | 99.76% | 99.98% |
| JPEG quality 90 | 50.49% | 98.24% |
| JPEG quality 70 | 50.48% | 92.08% |
| JPEG quality 50 | 50.50% | 86.24% |
| Resized to 224 pixels and back | 72.26% | 99.98% |
| Centre crop, not trained | 62.89% | 73.59% |

Before this training the model could not read the message after any JPEG compression. After it, the message survives JPEG and resizing well. Full-size photos do even better than these 128 pixel tests, because shrinking a photo before reading it smooths out most of the JPEG damage.

### Limits

- Our goal was more than 95% bit accuracy after JPEG. This is met at quality 90 in every test. At quality 70 and 50 it is met on full-size photos but not on small 128 pixel images.
- High bit accuracy does not always mean a readable verdict. After strong compression some photos still have a few wrong bits, and the checksum then rejects the message.
- The stamp is not fully invisible. Faint patterns can appear in smooth areas such as sky and near the image edges.
- Cropping was not part of training, and the stamp survives it poorly.

## Training

All models were trained on an NVIDIA RTX 2070 laptop GPU with 8 GB of memory, using PyTorch 2.4.1. Training took about 4 hours for the first notebook, 4 hours 40 minutes for the second and 8 hours for the third.

The data is from COCO 2017: 10,000 training photos from train2017, and 500 validation and 500 test photos from val2017. The first two notebooks train for 60 epochs. The third continues from the second for another 100 epochs with a lower learning rate.

## Sources

- Zhu, Kaplan, Johnson and Fei-Fei, [HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937), ECCV 2018.
- ando-khachatryan, [PyTorch implementation of HiDDeN](https://github.com/ando-khachatryan/HiDDeN), MIT license. The license text is in notebook 1.
- Tancik, Mildenhall and Ng, [StegaStamp: Invisible Hyperlinks in Physical Photographs](https://arxiv.org/abs/1904.05343), CVPR 2020.
- Jia, Fang and Zhang, [MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression](https://arxiv.org/abs/2108.08211), ACM MM 2021.
- [COCO dataset](https://cocodataset.org/).
