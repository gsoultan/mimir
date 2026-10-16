package parquet

import (
	"fmt"
)

// The enums of parquet.thrift that the reader interprets.

type encoding int32

const (
	encPlain                encoding = 0
	encPlainDictionary      encoding = 2
	encRLE                  encoding = 3
	encBitPacked            encoding = 4
	encDeltaBinaryPacked    encoding = 5
	encDeltaLengthByteArray encoding = 6
	encDeltaByteArray       encoding = 7
	encRLEDictionary        encoding = 8
	encByteStreamSplit      encoding = 9
)

func (e encoding) String() string {
	switch e {
	case encPlain:
		return "PLAIN"
	case encPlainDictionary:
		return "PLAIN_DICTIONARY"
	case encRLE:
		return "RLE"
	case encBitPacked:
		return "BIT_PACKED"
	case encDeltaBinaryPacked:
		return "DELTA_BINARY_PACKED"
	case encDeltaLengthByteArray:
		return "DELTA_LENGTH_BYTE_ARRAY"
	case encDeltaByteArray:
		return "DELTA_BYTE_ARRAY"
	case encRLEDictionary:
		return "RLE_DICTIONARY"
	case encByteStreamSplit:
		return "BYTE_STREAM_SPLIT"
	}
	return fmt.Sprintf("encoding %d", int32(e))
}

type codec int32

const (
	codecUncompressed codec = 0
	codecSnappy       codec = 1
	codecGzip         codec = 2
	codecLZO          codec = 3
	codecBrotli       codec = 4
	codecLZ4          codec = 5
	codecZstd         codec = 6
	codecLZ4Raw       codec = 7
)

func (c codec) String() string {
	switch c {
	case codecUncompressed:
		return "UNCOMPRESSED"
	case codecSnappy:
		return "SNAPPY"
	case codecGzip:
		return "GZIP"
	case codecLZO:
		return "LZO"
	case codecBrotli:
		return "BROTLI"
	case codecLZ4:
		return "LZ4"
	case codecZstd:
		return "ZSTD"
	case codecLZ4Raw:
		return "LZ4_RAW"
	}
	return fmt.Sprintf("codec %d", int32(c))
}

type pageType int32

const (
	dataPage       pageType = 0
	indexPage      pageType = 1
	dictionaryPage pageType = 2
	dataPageV2     pageType = 3
)

type repetition int32

const (
	repRequired repetition = 0
	repOptional repetition = 1
	repRepeated repetition = 2
)

// The structs of parquet.thrift, with the fields the reader uses. Unknown
// and unused fields are skipped, so files from newer writers still open.

type fileMetaData struct {
	version   int32
	schema    []schemaElement
	numRows   int64
	rowGroups []rowGroup
	keyValue  map[string]string
	createdBy string
	encrypted bool // encryption_algorithm is set
}

type schemaElement struct {
	typ          Type
	hasType      bool
	typeLength   int32
	repetition   repetition
	name         string
	numChildren  int32
	converted    convertedType
	hasConverted bool
	scale        int32 // of a DECIMAL converted type
	precision    int32
	logical      logicalType
}

// convertedType is the ConvertedType enum of parquet.thrift, the logical
// type annotation of older writers, which newer ones still write next to
// the LogicalType.
type convertedType int32

// convertedNames maps each converted type to the name of the logical type
// it stands for.
var convertedNames = [...]string{
	0:  "STRING", // UTF8
	1:  "MAP",
	2:  "MAP", // MAP_KEY_VALUE
	3:  "LIST",
	4:  "ENUM",
	5:  "DECIMAL",
	6:  "DATE",
	7:  "TIME", // TIME_MILLIS
	8:  "TIME", // TIME_MICROS
	9:  "TIMESTAMP",
	10: "TIMESTAMP",
	11: "UINT", // UINT_8 to UINT_64
	12: "UINT",
	13: "UINT",
	14: "UINT",
	15: "INT", // INT_8 to INT_64
	16: "INT",
	17: "INT",
	18: "INT",
	19: "JSON",
	20: "BSON",
	21: "INTERVAL",
}

// logicalType is the LogicalType union of parquet.thrift, reduced to the
// name of the member that is set and the parameters of a DECIMAL.
type logicalType struct {
	name             string
	scale, precision int32
}

// logicalNames maps the field ids of the LogicalType union to their names.
// INTEGER (10) is "INT" or "UINT" by its sign; UNKNOWN (11), the type of
// a column of nulls only, and members newer than FLOAT16 have no name.
var logicalNames = map[int16]string{
	1:  "STRING",
	2:  "MAP",
	3:  "LIST",
	4:  "ENUM",
	5:  "DECIMAL",
	6:  "DATE",
	7:  "TIME",
	8:  "TIMESTAMP",
	12: "JSON",
	13: "BSON",
	14: "UUID",
	15: "FLOAT16",
}

type rowGroup struct {
	columns       []columnChunk
	totalByteSize int64
	numRows       int64
}

type columnChunk struct {
	filePath  string
	meta      columnMetaData
	hasMeta   bool
	encrypted bool // crypto_metadata or encrypted_column_metadata is set
}

type columnMetaData struct {
	typ                   Type
	encodings             []encoding
	path                  []string
	codec                 codec
	numValues             int64
	totalUncompressedSize int64
	totalCompressedSize   int64
	dataPageOffset        int64
	indexPageOffset       int64
	dictionaryPageOffset  int64
	hasDictionaryOffset   bool
}

type pageHeader struct {
	typ              pageType
	uncompressedSize int32
	compressedSize   int32
	data             dataPageHeader
	hasData          bool
	dict             dictionaryPageHeader
	hasDict          bool
	v2               dataPageHeaderV2
	hasV2            bool
}

type dataPageHeader struct {
	numValues   int32
	encoding    encoding
	defEncoding encoding
	repEncoding encoding
}

type dictionaryPageHeader struct {
	numValues int32
	encoding  encoding
	isSorted  bool
}

type dataPageHeaderV2 struct {
	numValues    int32
	numNulls     int32
	numRows      int32
	encoding     encoding
	defLength    int32
	repLength    int32
	isCompressed bool
}

func parseFileMetaData(b []byte) (fileMetaData, error) {
	d := &decoder{buf: b}
	var m fileMetaData
	err := d.strct(func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			m.version, err = d.readI32(typ)
		case 2:
			n, err := d.readList(typ, ttStruct)
			if err != nil {
				return err
			}
			m.schema = make([]schemaElement, n)
			for i := range m.schema {
				if err := d.strct(m.schema[i].parse(d)); err != nil {
					return err
				}
			}
		case 3:
			m.numRows, err = d.readI64(typ)
		case 4:
			n, err := d.readList(typ, ttStruct)
			if err != nil {
				return err
			}
			m.rowGroups = make([]rowGroup, n)
			for i := range m.rowGroups {
				if err := d.strct(m.rowGroups[i].parse(d)); err != nil {
					return err
				}
			}
		case 5:
			m.keyValue, err = parseKeyValues(d, typ)
		case 6:
			m.createdBy, err = d.readString(typ)
		case 8:
			m.encrypted = true
			err = d.skip(typ)
		default:
			err = d.skip(typ)
		}
		return err
	})
	if err != nil {
		return m, err
	}
	if d.pos != len(b) {
		return m, fmt.Errorf("%d trailing bytes", len(b)-d.pos)
	}
	return m, nil
}

func parseKeyValues(d *decoder, typ byte) (map[string]string, error) {
	n, err := d.readList(typ, ttStruct)
	if err != nil {
		return nil, err
	}
	kv := make(map[string]string, n)
	for range n {
		var key, value string
		err := d.strct(func(id int16, typ byte) (err error) {
			switch id {
			case 1:
				key, err = d.readString(typ)
			case 2:
				value, err = d.readString(typ)
			default:
				err = d.skip(typ)
			}
			return err
		})
		if err != nil {
			return nil, err
		}
		kv[key] = value
	}
	return kv, nil
}

func (e *schemaElement) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			v, err := d.readI32(typ)
			if err != nil {
				return err
			}
			if v < 0 || v > 255 {
				return fmt.Errorf("physical type %d", v)
			}
			e.typ, e.hasType = Type(v), true
		case 2:
			e.typeLength, err = d.readI32(typ)
		case 3:
			v, err := d.readI32(typ)
			if err != nil {
				return err
			}
			if v < 0 || v > 2 {
				return fmt.Errorf("repetition type %d", v)
			}
			e.repetition = repetition(v)
		case 4:
			e.name, err = d.readString(typ)
		case 5:
			e.numChildren, err = d.readI32(typ)
		case 6:
			v, err := d.readI32(typ)
			if err != nil {
				return err
			}
			e.converted, e.hasConverted = convertedType(v), true
		case 7:
			e.scale, err = d.readI32(typ)
		case 8:
			e.precision, err = d.readI32(typ)
		case 10:
			err = d.readStruct(typ, e.logical.parse(d))
		default:
			err = d.skip(typ)
		}
		return err
	}
}

// parse reads the member of the LogicalType union that is set.
func (l *logicalType) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) error {
		switch id {
		case 5: // DecimalType
			l.name = "DECIMAL"
			return d.readStruct(typ, func(id int16, typ byte) (err error) {
				switch id {
				case 1:
					l.scale, err = d.readI32(typ)
				case 2:
					l.precision, err = d.readI32(typ)
				default:
					err = d.skip(typ)
				}
				return err
			})
		case 10: // IntType
			l.name = "INT"
			return d.readStruct(typ, func(id int16, typ byte) error {
				if id != 2 {
					return d.skip(typ)
				}
				signed, err := d.readBool(typ)
				if err == nil && !signed {
					l.name = "UINT"
				}
				return err
			})
		}
		l.name = logicalNames[id]
		return d.skip(typ)
	}
}

// logicalType returns the name of the element's logical type, from the
// LogicalType when it names one and from the converted type otherwise,
// with the precision and scale of a DECIMAL.
func (e *schemaElement) logicalType() (name string, precision, scale int32) {
	switch {
	case e.logical.name != "":
		return e.logical.name, e.logical.precision, e.logical.scale
	case e.hasConverted && e.converted >= 0 && int(e.converted) < len(convertedNames):
		return convertedNames[e.converted], e.precision, e.scale
	}
	return "", 0, 0
}

func (g *rowGroup) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			n, err := d.readList(typ, ttStruct)
			if err != nil {
				return err
			}
			g.columns = make([]columnChunk, n)
			for i := range g.columns {
				if err := d.strct(g.columns[i].parse(d)); err != nil {
					return err
				}
			}
		case 2:
			g.totalByteSize, err = d.readI64(typ)
		case 3:
			g.numRows, err = d.readI64(typ)
		default:
			err = d.skip(typ)
		}
		return err
	}
}

func (c *columnChunk) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			c.filePath, err = d.readString(typ)
		case 3:
			c.hasMeta = true
			err = d.readStruct(typ, c.meta.parse(d))
		case 8, 9:
			c.encrypted = true
			err = d.skip(typ)
		default:
			err = d.skip(typ)
		}
		return err
	}
}

func (m *columnMetaData) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			v, err := d.readI32(typ)
			if err != nil {
				return err
			}
			if v < 0 || v > 255 {
				return fmt.Errorf("physical type %d", v)
			}
			m.typ = Type(v)
		case 2:
			n, err := d.readList(typ, ttI32)
			if err != nil {
				return err
			}
			m.encodings = make([]encoding, n)
			for i := range m.encodings {
				v, err := d.i32()
				if err != nil {
					return err
				}
				m.encodings[i] = encoding(v)
			}
		case 3:
			n, err := d.readList(typ, ttBinary)
			if err != nil {
				return err
			}
			m.path = make([]string, n)
			for i := range m.path {
				b, err := d.binary()
				if err != nil {
					return err
				}
				m.path[i] = string(b)
			}
		case 4:
			v, err := d.readI32(typ)
			if err != nil {
				return err
			}
			m.codec = codec(v)
		case 5:
			m.numValues, err = d.readI64(typ)
		case 6:
			m.totalUncompressedSize, err = d.readI64(typ)
		case 7:
			m.totalCompressedSize, err = d.readI64(typ)
		case 9:
			m.dataPageOffset, err = d.readI64(typ)
		case 10:
			m.indexPageOffset, err = d.readI64(typ)
		case 11:
			m.dictionaryPageOffset, err = d.readI64(typ)
			m.hasDictionaryOffset = true
		default:
			err = d.skip(typ)
		}
		return err
	}
}

// parsePageHeader parses a page header from the start of b into h and
// returns the number of bytes it took.
func parsePageHeader(b []byte, h *pageHeader) (int, error) {
	d := &decoder{buf: b}
	*h = pageHeader{}
	err := d.strct(func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			v, err := d.readI32(typ)
			h.typ = pageType(v)
			return err
		case 2:
			h.uncompressedSize, err = d.readI32(typ)
		case 3:
			h.compressedSize, err = d.readI32(typ)
		case 5:
			h.hasData = true
			err = d.readStruct(typ, h.data.parse(d))
		case 7:
			h.hasDict = true
			err = d.readStruct(typ, h.dict.parse(d))
		case 8:
			h.hasV2 = true
			err = d.readStruct(typ, h.v2.parse(d))
		default:
			err = d.skip(typ)
		}
		return err
	})
	return d.pos, err
}

func (h *dataPageHeader) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		var v int32
		switch id {
		case 1:
			h.numValues, err = d.readI32(typ)
		case 2:
			v, err = d.readI32(typ)
			h.encoding = encoding(v)
		case 3:
			v, err = d.readI32(typ)
			h.defEncoding = encoding(v)
		case 4:
			v, err = d.readI32(typ)
			h.repEncoding = encoding(v)
		default:
			err = d.skip(typ)
		}
		return err
	}
}

func (h *dictionaryPageHeader) parse(d *decoder) func(id int16, typ byte) error {
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			h.numValues, err = d.readI32(typ)
		case 2:
			v, err := d.readI32(typ)
			h.encoding = encoding(v)
			return err
		case 3:
			h.isSorted, err = d.readBool(typ)
		default:
			err = d.skip(typ)
		}
		return err
	}
}

func (h *dataPageHeaderV2) parse(d *decoder) func(id int16, typ byte) error {
	h.isCompressed = true // the thrift default
	return func(id int16, typ byte) (err error) {
		switch id {
		case 1:
			h.numValues, err = d.readI32(typ)
		case 2:
			h.numNulls, err = d.readI32(typ)
		case 3:
			h.numRows, err = d.readI32(typ)
		case 4:
			v, err := d.readI32(typ)
			h.encoding = encoding(v)
			return err
		case 5:
			h.defLength, err = d.readI32(typ)
		case 6:
			h.repLength, err = d.readI32(typ)
		case 7:
			h.isCompressed, err = d.readBool(typ)
		default:
			err = d.skip(typ)
		}
		return err
	}
}
