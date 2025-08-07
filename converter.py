import os
import struct
import numpy as np

def size_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def read_tensor(filename):
    with open(filename, 'rb') as f:
        def unpack(fmt):
            result = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            return result if len(result) > 1 else result[0]

        if f.read(12) != 'tensor_file\0'.encode('utf8'):
            raise Exception('Invalid tensor file (header not recognized)')

        if unpack('<BB') != (1, 0):
            raise Exception('Invalid tensor file (unrecognized '
                            'file format version)')

        field_count = unpack('<I')
        size = os.stat(filename).st_size
        print('Loading tensor data from \"%s\" .. (%s, %i field%s)'
            % (filename, size_fmt(size),
               field_count, 's' if field_count > 1 else ''))

        # Maps from Struct.EType field in Mitsuba
        dtype_map = {
            1: np.uint8,
            2: np.int8,
            3: np.uint16,
            4: np.int16,
            5: np.uint32,
            6: np.int32,
            7: np.uint64,
            8: np.int64,
            9: np.float16,
            10: np.float32,
            11: np.float64
        }

        fields = {}
        for i in range(field_count):
            field_name = f.read(unpack('<H')).decode('utf8')
            field_ndim = unpack('<H')
            field_dtype = dtype_map[unpack('<B')]
            field_offset = unpack('<Q')
            field_shape = unpack('<' + 'Q' * field_ndim)
            fields[field_name] = (field_offset, field_dtype, field_shape)

        result = {}
        for k, v in fields.items():
            f.seek(v[0])
            result[k] = np.fromfile(f, dtype=v[1],
                                    count=np.prod(v[2])).reshape(v[2])
    return result


def write_tensor(filename, align=8, **kwargs):
    with open(filename, 'wb') as f:
        # Identifier
        f.write('tensor_file\0'.encode('utf8'))

        # Version number
        f.write(struct.pack('<BB', 1, 0))

        # Number of fields
        f.write(struct.pack('<I', len(kwargs)))

        # Maps to Struct.EType field in Mitsuba
        dtype_map = {
            np.uint8: 1,
            np.int8: 2,
            np.uint16: 3,
            np.int16: 4,
            np.uint32: 5,
            np.int32: 6,
            np.uint64: 7,
            np.int64: 8,
            np.float16: 9,
            np.float32: 10,
            np.float64: 11
        }

        offsets = {}
        fields = dict(kwargs)

        # Write all fields
        for k, v in fields.items():
            if type(v) is str:
                v = np.frombuffer(v.encode('utf8'), dtype=np.uint8)
            else:
                v = np.ascontiguousarray(v)
            fields[k] = v

            # Field identifier
            label = k.encode('utf8')
            f.write(struct.pack('<H', len(label)))
            f.write(label)

            # Field dimension
            f.write(struct.pack('<H', v.ndim))

            found = False
            for dt in dtype_map.keys():
                if dt == v.dtype:
                    found = True
                    f.write(struct.pack('B', dtype_map[dt]))
                    break
            if not found:
                raise Exception("Unsupported dtype: %s" % str(v.dtype))

            # Field offset (unknown for now)
            offsets[k] = f.tell()
            f.write(struct.pack('<Q', 0))

            # Field sizes
            f.write(struct.pack('<' + ('Q' * v.ndim), *v.shape))

        for k, v in fields.items():
            # Set field offset
            pos = f.tell()

            # Pad to requested alignment
            pos = (pos + align - 1) // align * align

            f.seek(offsets[k])
            f.write(struct.pack('<Q', pos))
            f.seek(pos)

            # Field data
            v.tofile(f)

        print('Wrote \"%s\" (%s)' % (filename, size_fmt(f.tell())))

def my_read_tensor(filename, dtype):
    """
    Reads a tensor file and returns the data as a dictionary.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist.")

    with open(filename, 'rb') as f:
        def unpack(fmt):
            result = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
            return result if len(result) > 1 else result[0]

        # Identifier
        identifier = unpack('3c')
        version = unpack('I')
        nb_dim = unpack('P')

        # Read each dimension (size_t per dimension)
        dim_sizes = []
        for _ in range(nb_dim):
            dim_sizes.append(unpack('P'))

        # Compute total number of elements
        total_elements = np.prod(dim_sizes)

        # Read the actual data
        data = np.frombuffer(f.read(total_elements * (8 if dtype == np.float64 else 4)), dtype=dtype)

        # Reshape data to the dimensions read
        data = data.reshape(dim_sizes)

        return data

sky_rad_rgb = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sky_rgb_rad.bin', np.float64)
sky_params_rgb = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sky_rgb_params.bin', np.float64)
sky_rad_spec = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sky_spec_rad.bin', np.float64)
sky_params_spec = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sky_spec_params.bin', np.float64)

sun_rad_rgb = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sun_rgb_rad.bin', np.float64)
sun_rad_spec = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sun_spec_rad.bin', np.float64)
sun_ld_spec = my_read_tensor('../mitsuba3/resources/data/sunsky/output/sun_spec_ld.bin', np.float64)


tgmm_tables = my_read_tensor('../mitsuba3/resources/data/sunsky/output/tgmm_tables.bin', np.float32)
write_tensor('datasets/sunsky_datasets.bin', 
             sky_rad_rgb=sky_rad_rgb, sky_params_rgb=sky_params_rgb, sky_rad_spec=sky_rad_spec, sky_params_spec=sky_params_spec, 
             sun_rad_rgb=sun_rad_rgb, sun_rad_spec=sun_rad_spec, sun_ld_spec=sun_ld_spec, 
)
