#!/usr/bin/env python3
# Writes the JPEG fixtures nvjpeg_paths.cpp decodes: one synthetic picture
# (gradients, a checkerboard, odd 61x45 size so every edge block is partial)
# saved by libjpeg through Pillow as baseline 4:4:4, 4:2:2 and 4:2:0,
# progressive 4:2:0 and 4:4:4, 4:2:0 with restart markers, grey baseline and
# progressive, and CMYK (Adobe, inverted); and a 333x251 noisy picture at
# quality 5 and as progressive 4:2:0 with restart markers. Same quality and so
# the same quantisation throughout, which is why the baseline and progressive
# renditions decode to identical pixels.
#
# Pillow 10.4 with libjpeg 6.2 (libjpeg-turbo) wrote the committed files.
import math
import random
from PIL import Image

W, H = 61, 45
im = Image.new('RGB', (W, H))
px = im.load()
for y in range(H):
    for x in range(W):
        r = int(127 + 120 * math.sin(x / 7.0))
        g = int(127 + 120 * math.cos(y / 5.0))
        b = int((x * 4 + y * 3) % 256)
        if (x // 8 + y // 8) % 2 == 0:
            r = 255 - r
        px[x, y] = (r, g, b)
im.save('b444.jpg', quality=85, subsampling=0)
im.save('b422.jpg', quality=85, subsampling=1)
im.save('b420.jpg', quality=85, subsampling=2)
im.save('p420.jpg', quality=85, subsampling=2, progressive=True)
im.save('p444.jpg', quality=85, subsampling=0, progressive=True)
im.save('r420.jpg', quality=85, subsampling=2, restart_marker_blocks=3)
im.convert('L').save('gray.jpg', quality=85)
im.convert('L').save('pgray.jpg', quality=85, progressive=True)
cmyk = Image.new('CMYK', (W, H))
cp = cmyk.load()
for y in range(H):
    for x in range(W):
        cp[x, y] = ((x * 4) % 256, (y * 5) % 256, (x * y) % 256, (x * 3 + y * 2) % 256)
cmyk.save('cmyk2.jpg', quality=95)

random.seed(7)
W, H = 333, 251
big = Image.new('RGB', (W, H))
bp = big.load()
for y in range(H):
    for x in range(W):
        v = int(128 + 60 * math.sin(x / 13.0 + y / 29.0) + 40 * math.cos(x * y / 900.0))
        bp[x, y] = (max(0, min(255, v + random.randint(-30, 30))), max(0, min(255, 255 - v + random.randint(-20, 20))),
                    (x ^ y) & 255)
big.save('big_q5.jpg', quality=5, subsampling=2)
big.save('big_pr.jpg', quality=75, subsampling=2, progressive=True, restart_marker_rows=2)
