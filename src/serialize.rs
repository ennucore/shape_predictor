use Vector2;
use nalgebra::{Dim, MatrixVec};
use nalgebra::core::dimension::Dynamic;
use {ShapePredictor, Matrix, RegressionTree, SplitFeature};

#[derive(Serialize, Deserialize)]
struct MatrixSerialize {
    ncols: usize,
    nrows: usize,
    vec: Vec<f32>
}

impl MatrixSerialize {
    fn from(matrix: &Matrix) -> Self {
        Self {
            ncols: matrix.ncols(),
            nrows: matrix.nrows(),
            vec: matrix.data.data().clone()
        }
    }

    fn to(&self) -> Matrix {
        let data = MatrixVec::new(Dynamic::from_usize(self.ncols), Dynamic::from_usize(self.nrows), self.vec.clone());
        Matrix::from_data(data)
    }
}

#[derive(Serialize, Deserialize)]
struct RegressionTreeSerialize {
    splits: Vec<SplitFeature>,
    leaf_values: Vec<MatrixSerialize>
}

impl RegressionTreeSerialize {
    fn from(tree: &RegressionTree) -> Self {
        Self {
            splits: tree.splits.clone(),
            leaf_values: tree.leaf_values.iter().map(MatrixSerialize::from).collect()
        }
    }

    fn to(&self) -> RegressionTree {
        RegressionTree {
            splits: self.splits.clone(),
            leaf_values: self.leaf_values.iter().map(MatrixSerialize::to).collect()
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct ShapePredictorSerialize {
    initial_shape: MatrixSerialize,
    forests: Vec<Vec<RegressionTreeSerialize>>,
    deltas: Vec<Vec<Vector2Serialize>>,
    anchor_idx: Vec<Vec<u64>>
}

impl ShapePredictorSerialize {
    pub fn from(shape: &ShapePredictor) -> Self {
        Self {
            initial_shape: MatrixSerialize::from(&shape.initial_shape),
            forests: shape.forests.iter().map(|forest| {
                forest.iter().map(RegressionTreeSerialize::from).collect()
            }).collect(),
            deltas: shape.deltas.iter().map(|delta| {
                delta.iter().map(Vector2Serialize::from).collect()
            }).collect(),
            anchor_idx: shape.anchor_idx.clone()
        }
    }

    pub fn to(self) -> ShapePredictor {
        ShapePredictor {
            initial_shape: MatrixSerialize::to(&self.initial_shape),
            forests: self.forests.iter().map(|forest| {
                forest.iter().map(RegressionTreeSerialize::to).collect()
            }).collect(),
            deltas: self.deltas.iter().map(|delta| {
                delta.iter().map(Vector2Serialize::to).collect()
            }).collect(),
            anchor_idx: self.anchor_idx
        }
    }
}

#[derive(Serialize, Deserialize)]
struct Vector2Serialize {
    x: f32,
    y: f32
}

impl Vector2Serialize {
    fn from(vector2: &Vector2) -> Self {
        Self {
            x: vector2.x,
            y: vector2.y
        }
    }

    fn to(&self) -> Vector2 {
        Vector2::new(self.x, self.y)
    }
}