#!/usr/bin/env python3
"""Render a GIF of seam carving an image down to 50% of its width and height.

Timeline (10 s by default):
  1. shrink width by 50%
  2. rotate clockwise by 90 degrees
  3. shrink width (former height) by 50%
  4. rotate back counterclockwise

Build the extension first (./build.sh), then run:
  python shrink_animation.py [input.jpeg] [-o output.gif]
"""
import argparse
import time
from pathlib import Path

from PIL import Image

from cpp_seam_carver import CppSeamCarver

FRAME_MS = 100
BACKGROUND = (24, 24, 24)


def shrink_frames(image, frame_count):
    """Yield frame_count images, ending with image shrunk to 50% of its width."""
    carver = CppSeamCarver(image)
    width = image.width
    target_width = width // 2
    for i in range(1, frame_count + 1):
        frame_width = width - round(i * (width - target_width) / frame_count)
        carver.shrink_horizontally(carver.width - frame_width, True)
        yield carver.get_shrunk_image()


def rotate_frames(image, frame_count, clockwise):
    """Yield frame_count images rotating image by 90 degrees; the last one is exact."""
    sign = -1 if clockwise else 1  # PIL rotates counterclockwise for positive angles
    for i in range(1, frame_count):
        yield image.rotate(sign * 90 * i / frame_count, resample=Image.BICUBIC,
                           expand=True, fillcolor=BACKGROUND)
    yield image.transpose(Image.ROTATE_270 if clockwise else Image.ROTATE_90)


def build_timeline(image, duration_ms):
    """Return a list of (image, duration_ms) frames."""
    # Share of the total duration for each phase
    plan = [('hold', 0.05), ('shrink', 0.35), ('rotate_cw', 0.10),
            ('shrink', 0.30), ('rotate_ccw', 0.10), ('hold', 0.10)]
    frames = []
    current = image.convert('RGB')
    for phase, share in plan:
        phase_ms = round(duration_ms * share / 10) * 10  # GIF delays are in 10 ms units
        if phase == 'hold':
            frames.append((current, phase_ms))
            continue
        frame_count = max(1, phase_ms // FRAME_MS)
        if phase == 'shrink':
            phase_images = list(shrink_frames(current, frame_count))
        else:
            phase_images = list(rotate_frames(current, frame_count, phase == 'rotate_cw'))
        frames.extend((frame, FRAME_MS) for frame in phase_images)
        frames[-1] = (frames[-1][0], phase_ms - FRAME_MS * (frame_count - 1))
        current = phase_images[-1]
    return frames


def compose(frames, scale):
    """Center every frame on a common canvas and downscale it."""
    canvas_width = max(frame.width for frame, _ in frames)
    canvas_height = max(frame.height for frame, _ in frames)
    out_size = (round(canvas_width * scale), round(canvas_height * scale))
    composed = []
    for frame, duration in frames:
        canvas = Image.new('RGB', (canvas_width, canvas_height), BACKGROUND)
        canvas.paste(frame, ((canvas_width - frame.width) // 2, (canvas_height - frame.height) // 2))
        composed.append((canvas.resize(out_size, Image.BILINEAR), duration))
    return composed


def main():
    default_input = Path(__file__).resolve().parent.parent / 'tundra_1280x958.jpeg'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', nargs='?', type=Path, default=default_input)
    parser.add_argument('-o', '--output', type=Path, default=Path('tundra_shrink_animation.gif'))
    parser.add_argument('--duration', type=float, default=10.0, help='animation length in seconds')
    parser.add_argument('--scale', type=float, default=0.4, help='output scale of the GIF frames')
    args = parser.parse_args()

    start = time.perf_counter()
    frames = build_timeline(Image.open(args.input), round(args.duration * 1000))
    carved = time.perf_counter()
    frames = compose(frames, args.scale)
    images = [frame for frame, _ in frames]
    images[0].save(args.output, save_all=True, append_images=images[1:],
                   duration=[duration for _, duration in frames], loop=0)
    done = time.perf_counter()

    total_ms = sum(duration for _, duration in frames)
    print(f'{len(frames)} frames, {total_ms / 1000:.1f} s, {images[0].width}x{images[0].height} -> {args.output}')
    print(f'carving {carved - start:.1f} s, encoding {done - carved:.1f} s')


if __name__ == '__main__':
    main()
