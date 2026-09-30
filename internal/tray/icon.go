package tray

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/color"
	"image/png"
)

// Icon returns a 32x32 .ico (PNG payload, supported since Windows Vista):
// the control panel's accent-blue rounded square with a white "Q".
func Icon() []byte {
	const n = 32
	img := image.NewNRGBA(image.Rect(0, 0, n, n))
	blue := color.NRGBA{0x5b, 0x8c, 0xff, 0xff}
	white := color.NRGBA{0xff, 0xff, 0xff, 0xff}

	inRounded := func(x, y int) bool {
		const r = 7
		cx, cy := x, y
		switch {
		case x < r && y < r:
			cx, cy = r, r
		case x >= n-r && y < r:
			cx, cy = n-r-1, r
		case x < r && y >= n-r:
			cx, cy = r, n-r-1
		case x >= n-r && y >= n-r:
			cx, cy = n-r-1, n-r-1
		default:
			return true
		}
		dx, dy := x-cx, y-cy
		return dx*dx+dy*dy <= r*r
	}
	for y := 0; y < n; y++ {
		for x := 0; x < n; x++ {
			if !inRounded(x, y) {
				continue
			}
			img.Set(x, y, blue)
			// "Q": a ring plus a short diagonal tail
			dx, dy := x-15, y-14
			d := dx*dx + dy*dy
			ring := d <= 9*9 && d >= 6*6
			tail := x >= 17 && x <= 25 && y >= 19 && y <= 27 && (x-y >= -3 && x-y <= 0)
			if ring || tail {
				img.Set(x, y, white)
			}
		}
	}

	var pngBuf bytes.Buffer
	if err := png.Encode(&pngBuf, img); err != nil {
		panic(err) // encoding an in-memory NRGBA image cannot fail
	}

	var ico bytes.Buffer
	le := binary.LittleEndian
	binary.Write(&ico, le, [3]uint16{0, 1, 1}) // reserved, type=icon, count
	binary.Write(&ico, le, struct {
		W, H, Colors, Reserved uint8
		Planes, BPP            uint16
		Size, Offset           uint32
	}{n, n, 0, 0, 1, 32, uint32(pngBuf.Len()), 6 + 16})
	ico.Write(pngBuf.Bytes())
	return ico.Bytes()
}
