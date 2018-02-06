use nom::*;

use nalgebra::{Dim, MatrixVec};
use nalgebra::core::dimension::Dynamic;
use {SplitFeature, RegressionTree, Matrix, ShapePredictor, Error, Vector2};

// https://github.com/davisking/dlib/blob/master/dlib/serialize.h#L288
named!(dlib_int<i64>, do_parse!(
    control_byte: le_u8 >>
    buf: take!((control_byte & 0x0F) as usize) >>
    ({
        let size = (control_byte & 0x0F) as usize;
        let mut int = 0;

        for i in 0 .. size {
            int <<= 8;
            int |= i64::from(buf[size - 1 - i]);
        }

        if control_byte >> 7 == 1 {
            int *= -1;
        }

        int
    })
));

// https://github.com/davisking/dlib/blob/master/dlib/serialize.h#L134
// https://github.com/davisking/dlib/blob/master/dlib/float_details.h#L143
named!(dlib_float<f32>, do_parse!(
    mantissa: dlib_int >>
    exponent: dlib_int >>
    ((mantissa as f32) * 2.0_f32.powf(exponent as f32))
));

// https://github.com/davisking/dlib/blob/master/dlib/geometry/vector.h#L1133
named!(vector2<Vector2>, do_parse!(
    x: dlib_float >>
    y: dlib_float >>
    (Vector2::new(x, y))
));

// https://github.com/davisking/dlib/blob/master/dlib/image_processing/shape_predictor.h#L34
named!(split_feature<SplitFeature>, do_parse!(
    idx1: dlib_int >>
    idx2: dlib_int >>
    thresh: dlib_float >>
    (SplitFeature {
        idx1: idx1 as usize,
        idx2: idx2 as usize,
        thresh
    })
));

// https://github.com/davisking/dlib/blob/master/dlib/matrix/matrix.h#L1888
named!(matrix_dimensions<(usize, usize)>, do_parse!(
    rows: dlib_int >>
    cols: dlib_int >>
    (if rows < 0 || cols < 0 {
        (-rows as usize, -cols as usize)
    } else {
        (rows as usize, cols as usize)
    })
));

// https://github.com/davisking/dlib/blob/master/dlib/matrix/matrix.h#L1888
named!(matrix<Matrix>, do_parse!(
    dimensions: matrix_dimensions >>
    values: many_m_n!(dimensions.0 * dimensions.1, dimensions.0 * dimensions.1, dlib_float) >>
    ({
        let data = MatrixVec::new(Dynamic::from_usize(dimensions.0), Dynamic::from_usize(dimensions.1), values);
        Matrix::from_data(data)
    })
));

// https://github.com/davisking/dlib/blob/master/dlib/image_processing/shape_predictor.h#L96
named!(regression_tree<RegressionTree>, do_parse!(
    len: dlib_int >>
    splits: many_m_n!(len as usize, len as usize, split_feature) >>
    len: dlib_int >>
    leaf_values: many_m_n!(len as usize, len as usize, matrix) >>
    (RegressionTree {
        splits, leaf_values
    })
));

named!(anchor<u64>, do_parse!(
    anchor: dlib_int >>
    (anchor as u64)
));

// https://github.com/davisking/dlib/blob/master/dlib/image_processing/shape_predictor.h#L421
named!(version<i64>, do_parse!(
    version: dlib_int >>
    ({
        assert_eq!(version, 1);
        version
    })
));

// https://github.com/davisking/dlib/blob/master/dlib/image_processing/shape_predictor.h#L421
named!(shape_predictor<ShapePredictor>, do_parse!(
    version >>
    initial_shape: matrix >>
    len: dlib_int >>
    forests: many_m_n!(len as usize, len as usize, do_parse!(
        len: dlib_int >>
        trees: many_m_n!(len as usize, len as usize, regression_tree) >>
        (trees)
    )) >>
    len: dlib_int >>
    anchor_idx: many_m_n!(len as usize, len as usize, do_parse!(
        len: dlib_int >>
        anchors: many_m_n!(len as usize, len as usize, anchor) >>
        (anchors)
    )) >>
    len: dlib_int >>
    deltas: many_m_n!(len as usize, len as usize, do_parse!(
        len: dlib_int >>
        deltas: many_m_n!(len as usize, len as usize, vector2) >>
        (deltas)
    )) >>
    (ShapePredictor {
        initial_shape, forests, anchor_idx, deltas
    })
));

pub fn parse_shape_predictor(slice: &[u8]) -> Result<ShapePredictor, Error> {
    match shape_predictor(slice) {
        IResult::Done(_, predictor) => Ok(predictor),
        IResult::Incomplete(needed) => Err(Error::ReachedEof(needed)),
        IResult::Error(error) => Err(Error::ParsingError(error.into_error_kind()))
    }
}

#[cfg(test)]
mod tests {
    use dlib_parser::*;
    use std::io::Read;
    use std::fs::File;

    fn read_bytes(filename: &str) -> Vec<u8> {
        let mut vec = Vec::new();
        File::open(&format!("test_files/{}", filename)).unwrap().read_to_end(&mut vec).unwrap();
        println!("{:?}", vec);
        vec
    }

    #[test]
    fn float() {
        let bytes = read_bytes("float");
        let f = dlib_float(&bytes).unwrap().1;
        assert_eq!(f, 5.5);
    }

    #[test]
    fn vec2() {
        let bytes = read_bytes("vec2");
        let f = vector2(&bytes).unwrap().1;
        assert_eq!(f, Vector2::new(1.0, 3.0));
    }

    #[test]
    fn mat4x4() {
        let bytes = read_bytes("mat4x4");
        let f = matrix(&bytes).unwrap().1;
        assert_eq!(f.data.data().len(), 16);
    }

    #[test]
    fn predictor() {
        let bytes = read_bytes("../shape_predictor_68_face_landmarks.dat");
        shape_predictor(&bytes).unwrap();
    }
}