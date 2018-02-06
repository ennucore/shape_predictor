extern crate image;
extern crate nalgebra;
extern crate num_traits;
#[macro_use]
extern crate nom;
extern crate serde;
#[macro_use]
extern crate serde_derive;
extern crate bincode;

use nalgebra::{MatrixVec, Dynamic};
pub type Vector2 = nalgebra::Vector2<f32>;
type Matrix = nalgebra::Matrix<f32, Dynamic, Dynamic, MatrixVec<f32, Dynamic, Dynamic>>;

use image::{GenericImage, Pixel};
use num_traits::cast::NumCast;

mod transformations;
mod dlib_parser;
mod serialize;
use transformations::PointTransformationAffine;
use serialize::ShapePredictorSerialize;

use std::io::{self, BufReader, BufWriter, Read};
use std::fs::File;

#[derive(Debug)]
pub enum Error {
    ReachedEof(nom::Needed),
    ParsingError(nom::ErrorKind),
    Io(io::Error),
    Serialization(bincode::Error)
}

impl From<io::Error> for Error {
    fn from(error: io::Error) -> Self {
        Error::Io(error)
    }
}

impl From<bincode::Error> for Error {
    fn from(error: bincode::Error) -> Self {
        Error::Serialization(error)
    }
}

fn length_squared(vec: Vector2) -> f32 {
    // A^2 + B^2 == C^2
    vec.x.powi(2) + vec.y.powi(2)
}

fn get_pixel_intensity<I: GenericImage>(image: &I, pos: Vector2) -> f32 {
    let pixel = image.get_pixel(pos.x as u32, pos.y as u32);
    let value = pixel.to_luma().data[0];
    NumCast::from(value).unwrap()
}

fn location(shape: &Matrix, idx: u64) -> Vector2 {
    let idx = idx as usize;
    debug_assert!(idx < shape.len() / 2);
    debug_assert_eq!(shape.len() % 2, 0);

    Vector2::new(shape[idx * 2], shape[idx * 2 + 1])
}

pub struct ShapePredictor {
    initial_shape: Matrix,
    forests: Vec<Vec<RegressionTree>>,
    deltas: Vec<Vec<Vector2>>,
    anchor_idx: Vec<Vec<u64>>
}

impl ShapePredictor {
    // https://github.com/davisking/dlib/blob/master/dlib/image_processing/shape_predictor.h#L339
    /// Run the shape predictor on an image with a specific region of interest and get the positions of landmarks.
    pub fn run<I: GenericImage>(&self, image: &I, region: &Rectangle) -> Vec<Vector2> {
        let mut current_shape = self.initial_shape.clone();
        let mut feature_pixel_values = Vec::new();
        let tform_to_img = PointTransformationAffine::unnormalising(region);

        for iter in 0 .. self.forests.len() {
            self.extract_feature_pixel_values(image, region, &current_shape, iter, &mut feature_pixel_values);
            let mut leaf_idx = 0;

            for tree in &self.forests[iter] {
                current_shape += tree.find(&feature_pixel_values, &mut leaf_idx);
            }
        }

        (0 .. current_shape.len() / 2)
            .map(|i| tform_to_img.mul(location(&current_shape, i as u64)))
            .collect()
    }

    fn extract_feature_pixel_values<I: GenericImage>(
        &self, image: &I, region: &Rectangle, current_shape: &Matrix, iter: usize,
        feature_pixel_values: &mut Vec<f32>
    ) {
        let reference_pixel_anchor_idx = &self.anchor_idx[iter];
        let reference_pixel_deltas = &self.deltas[iter];

        debug_assert_eq!(reference_pixel_anchor_idx.len(), reference_pixel_deltas.len());
        debug_assert_eq!(current_shape.len(), self.initial_shape.len());
        debug_assert_eq!(self.initial_shape.len() % 2, 0);

        let tform = self.find_tform_between(current_shape).m;
        let tform_to_img = PointTransformationAffine::unnormalising(region);

        let area = Rectangle::from_image(image);

        *feature_pixel_values = (0 .. reference_pixel_deltas.len())
            .map(|i| {
                let point = tform_to_img.mul(tform * reference_pixel_deltas[i] + location(current_shape, reference_pixel_anchor_idx[i]));

                if area.contains(point) {
                    get_pixel_intensity(image, point)
                } else {
                    0.0
                }
            })
            .collect();
    }

    fn find_tform_between(&self, to_shape: &Matrix) -> PointTransformationAffine {
        debug_assert_eq!(self.initial_shape.len(), to_shape.len());
        debug_assert_eq!(self.initial_shape.len() % 2, 0);
        debug_assert!(!self.initial_shape.is_empty());

        let num = self.initial_shape.len() / 2;

        if num == 1 {
            PointTransformationAffine::default()
        } else {
            let mut from_points = Vec::with_capacity(num);
            let mut to_points = Vec::with_capacity(num);

            for i in 0 .. num as u64 {
                from_points.push(location(&self.initial_shape, i));
                to_points.push(location(to_shape, i));
            }

            PointTransformationAffine::find_similarity(&from_points, &to_points)
        }
    }

    /// Serialize the shape predictor to a file.
    pub fn write(&self, filename: &str) -> Result<(), Error> {
        let mut writer = BufWriter::new(File::create(filename)?);
        let serialize = ShapePredictorSerialize::from(self);
        bincode::serialize_into(&mut writer, &serialize, bincode::Infinite)?;
        Ok(())
    }

    /// Deserialize the shape predictor from a file.
    pub fn read(filename: &str) -> Result<Self, Error> {
        let mut reader = BufReader::new(File::open(filename)?);
        let deserialize: ShapePredictorSerialize = bincode::deserialize_from(&mut reader, bincode::Infinite)?;
        Ok(deserialize.to())
    }

    /// Deserialize the shape predictor from a file encoded by dlib.
    pub fn read_from_dlib(filename: &str) -> Result<Self, Error> {
        let mut buffer = Vec::new();
        File::open(filename)?.read_to_end(&mut buffer)?;
        dlib_parser::parse_shape_predictor(&buffer)
    }
}

#[derive(Deserialize, Serialize, Clone)]
struct SplitFeature {
    idx1: usize,
    idx2: usize,
    thresh: f32
}

struct RegressionTree {
    splits: Vec<SplitFeature>,
    leaf_values: Vec<Matrix>
}

impl RegressionTree {
    fn find<'a>(&'a self, feature_pixel_values: &[f32], i: &mut usize) -> &'a Matrix {
        debug_assert_eq!(self.leaf_values.len() % 2, 0);
        debug_assert_eq!(self.leaf_values.len(), self.splits.len() + 1);

        *i = 0;

        while *i < self.splits.len() {
            let split = &self.splits[*i];
            if feature_pixel_values[split.idx1] as f32 - feature_pixel_values[split.idx2] as f32 > split.thresh {
                *i = (2 * *i) + 1;
            } else {
                *i = (2 * *i) + 2;
            }
        }

        *i -= self.splits.len();
        &self.leaf_values[*i]
    }
}

/// A rectangle in the image.
#[derive(Debug)]
pub struct Rectangle {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32
}

impl Rectangle {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x, y, width, height
        }
    }

    pub fn tl_corner(&self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }

    pub fn tr_corner(&self) -> Vector2 {
        Vector2::new(self.x + self.width, self.y)
    }

    pub fn bl_corner(&self) -> Vector2 {
        Vector2::new(self.x, self.y + self.height)
    }

    pub fn br_corner(&self) -> Vector2 {
        Vector2::new(self.x + self.width, self.y + self.height)
    }

    fn contains(&self, vec: Vector2) -> bool {
        vec.x >= self.x && vec.y >= self.y && 
        vec.x <= (self.x + self.width) && vec.y <= (self.y + self.height)
    }

    /// Create a rectangle the same size as an image.
    pub fn from_image<I: GenericImage>(image: &I) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            width: image.width() as f32,
            height: image.height() as f32
        }
    }
}

#[test]
fn read() {
    ShapePredictor::read("wow.data").unwrap();
}