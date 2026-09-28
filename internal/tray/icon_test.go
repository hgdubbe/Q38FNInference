package tray

import (
	"bytes"
	"encoding/binary"
	"image/png"
	"testing"
)

func TestIconIsValidPNGIco(t *testing.T) {
	ico := Icon()
	var hdr [3]uint16
	binary.Read(bytes.NewReader(ico), binary.LittleEndian, &hdr)
	if hdr != [3]uint16{0, 1, 1} {
		t.Fatalf("ICO header = %v", hdr)
	}
	size := binary.LittleEndian.Uint32(ico[6+8:])
	offset := binary.LittleEndian.Uint32(ico[6+12:])
	if int(offset+size) != len(ico) {
		t.Fatalf("directory entry says %d+%d bytes, file has %d", offset, size, len(ico))
	}
	img, err := png.Decode(bytes.NewReader(ico[offset:]))
	if err != nil {
		t.Fatal(err)
	}
	if b := img.Bounds(); b.Dx() != 32 || b.Dy() != 32 {
		t.Errorf("icon is %v, want 32x32", b)
	}
}
