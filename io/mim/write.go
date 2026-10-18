package mim

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"io"

	mimfile "github.com/gsoultan/mimir-model"

	"github.com/gsoultan/mimir/tensor"
)

// fileTensor is a tensor that writeFile writes, laid out in the weights
// section.
type fileTensor struct {
	name   string
	t      *tensor.Tensor
	dtype  mimfile.DType
	shape  []uint64
	offset uint64 // from the start of the weights section
	length uint64
}

// fileSection is a section other than the weights that writeFile writes.
type fileSection struct {
	kind mimfile.SectionKind
	data []byte
}

// writeFile writes a file of the tensors ts, which are computed and
// sorted by name, and of the sections, as spec/container.md lays it out
// and mimfile.Writer would. Unlike mimfile.Writer, which holds the bytes
// of every tensor until it writes them, it copies each tensor's bytes
// from its device as it writes them, so a file takes as much memory as
// its largest tensor rather than the whole model. With hash, it computes
// the hash of the weights section first, which reads every tensor twice.
func writeFile(ctx context.Context, w io.Writer, ts []fileTensor, sections []fileSection, hashed bool) error {
	end := uint64(0)
	for i := range ts {
		t := &ts[i]
		align := uint64(mimfile.TensorAlignment)
		if t.length >= mimfile.LargeTensorThreshold {
			align = mimfile.LargeTensorAlignment
		}
		t.offset = alignUp(end, align)
		end = t.offset + t.length
	}
	dataLen := end

	var pool []byte
	for _, t := range ts {
		pool = append(pool, t.name...)
	}
	nsec := len(sections) + 1
	indexLen := uint64(mimfile.IndexHeaderSize + nsec*mimfile.SectionEntrySize + len(ts)*mimfile.TensorEntrySize + len(pool))
	if indexLen > mimfile.MaxIndexLength {
		return fmt.Errorf("index of %d bytes, more than %d", indexLen, mimfile.MaxIndexLength)
	}
	pos := alignUp(mimfile.PreambleSize+indexLen, sectionAlignment)
	offs := make([]uint64, len(sections))
	for i, s := range sections {
		offs[i] = pos
		pos = alignUp(pos+uint64(len(s.data)), sectionAlignment)
	}
	dataOff := alignUp(pos, mimfile.DefaultDataAlignment)

	hashes := make([][32]byte, len(sections))
	var weightsHash [32]byte
	if hashed {
		for i, s := range sections {
			hashes[i] = sha256.Sum256(s.data)
		}
		h := sha256.New()
		if err := writeWeights(ctx, &counter{w: h}, ts); err != nil {
			return err
		}
		h.Sum(weightsHash[:0])
	}

	le := binary.LittleEndian
	head := make([]byte, mimfile.PreambleSize+indexLen)
	copy(head, mimfile.Magic)
	le.PutUint16(head[4:], mimfile.FormatMajor)
	le.PutUint16(head[6:], mimfile.FormatMinor)
	le.PutUint32(head[8:], mimfile.ByteOrderMark)
	le.PutUint32(head[12:], mimfile.DefaultDataAlignment)
	le.PutUint64(head[16:], mimfile.PreambleSize)
	le.PutUint64(head[24:], indexLen)
	le.PutUint64(head[32:], dataOff)
	le.PutUint64(head[40:], dataLen)
	le.PutUint64(head[48:], dataOff+dataLen)
	idx := head[mimfile.PreambleSize:]
	le.PutUint32(idx[0:], uint32(nsec))
	le.PutUint32(idx[4:], uint32(len(ts)))
	le.PutUint64(idx[8:], uint64(len(pool)))
	entry := func(e []byte, kind mimfile.SectionKind, flags uint32, off, n uint64, h [32]byte) {
		le.PutUint32(e[0:], uint32(kind))
		le.PutUint32(e[4:], flags)
		le.PutUint64(e[8:], off)
		le.PutUint64(e[16:], n)
		if hashed {
			e[24] = byte(mimfile.HashSHA256)
			copy(e[32:], h[:])
		}
	}
	es := idx[mimfile.IndexHeaderSize:]
	entry(es, mimfile.SectionWeights, mimfile.FlagRequired, dataOff, dataLen, weightsHash)
	for i, s := range sections {
		entry(es[(i+1)*mimfile.SectionEntrySize:], s.kind, 0, offs[i], uint64(len(s.data)), hashes[i])
	}
	rows := idx[mimfile.IndexHeaderSize+nsec*mimfile.SectionEntrySize:]
	nameOff := 0
	for i, t := range ts {
		r := rows[i*mimfile.TensorEntrySize:]
		le.PutUint32(r[0:], uint32(nameOff))
		le.PutUint32(r[4:], uint32(len(t.name)))
		le.PutUint32(r[8:], uint32(t.dtype))
		r[12] = byte(len(t.shape))
		for d, v := range t.shape {
			le.PutUint64(r[16+8*d:], v)
		}
		le.PutUint64(r[80:], t.offset)
		le.PutUint64(r[88:], t.length)
		nameOff += len(t.name)
	}
	copy(rows[len(ts)*mimfile.TensorEntrySize:], pool)

	// The writes of padding are small, and a file of many small tensors
	// writes many of them.
	bw := bufio.NewWriterSize(w, 1<<20)
	c := &counter{w: bw}
	if _, err := c.Write(head); err != nil {
		return err
	}
	for i, s := range sections {
		if err := c.pad(offs[i]); err != nil {
			return err
		}
		if _, err := c.Write(s.data); err != nil {
			return err
		}
	}
	if err := c.pad(dataOff); err != nil {
		return err
	}
	if err := writeWeights(ctx, &counter{w: c}, ts); err != nil {
		return err
	}
	return bw.Flush()
}

// writeWeights writes the weights section, the bytes of ts at their
// offsets, to c, which counts from the section's start. The bytes of
// every tensor pass through one buffer.
func writeWeights(ctx context.Context, c *counter, ts []fileTensor) error {
	var largest uint64
	for _, t := range ts {
		largest = max(largest, t.length)
	}
	buf := make([]byte, largest)
	for _, t := range ts {
		if err := c.pad(t.offset); err != nil {
			return err
		}
		b := buf[:t.length]
		if err := t.t.ReadBytes(ctx, b); err != nil {
			return fmt.Errorf("%s: %w", t.name, err)
		}
		if _, err := c.Write(b); err != nil {
			return err
		}
	}
	return nil
}

// sectionAlignment is the alignment of the sections after the index, as
// mimfile.Writer places them: a multiple of mimfile.SectionAlignment.
const sectionAlignment = 64

func alignUp(n, a uint64) uint64 { return (n + a - 1) &^ (a - 1) }

// counter counts the bytes written through it.
type counter struct {
	w io.Writer
	n uint64
}

func (c *counter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += uint64(n)
	return n, err
}

var zeros [4096]byte

// pad writes zeros up to offset to.
func (c *counter) pad(to uint64) error {
	for c.n < to {
		if _, err := c.Write(zeros[:min(to-c.n, uint64(len(zeros)))]); err != nil {
			return err
		}
	}
	return nil
}
