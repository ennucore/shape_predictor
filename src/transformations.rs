use nalgebra::{Matrix, Matrix2, Matrix2x3, Dim, DimName, Dynamic, MatrixVec};
use nalgebra::storage::Storage;
use {length_squared, Rectangle, Vector2};

type ColVectorN<N> = Matrix<f32, N, Dynamic, MatrixVec<f32, N, Dynamic>>;

fn new_col_vector<N: Dim + DimName>(width: usize, height: usize) -> ColVectorN<N> {
    let data = MatrixVec::new(N::from_usize(width), Dynamic::new(height), vec![0.0; width * height]);
    ColVectorN::from_data(data)
}

pub struct PointTransformationAffine {
    pub m: Matrix2<f32>,
    b: Vector2
}

impl PointTransformationAffine {
    pub fn new(m: Matrix2<f32>, b: Vector2) -> Self {
        Self {
            m, b
        }
    }

    pub fn default() -> Self {
        Self {
            m: Matrix2::identity(),
            b: Vector2::new(0.0, 0.0)
        }
    }

    pub fn mul(&self, p: Vector2) -> Vector2 {
        self.m * p + self.b
    }

    pub fn unnormalising(rectangle: &Rectangle) -> Self {
        Self::find_affine(&[
            Vector2::new(0.0, 0.0),
            Vector2::new(1.0, 0.0),
            Vector2::new(1.0, 1.0)
        ], &[
            rectangle.tl_corner(),
            rectangle.tr_corner(),
            rectangle.br_corner()
        ])
    }

    pub fn find_affine(from_points: &[Vector2], to_points: &[Vector2]) -> Self {
        debug_assert_eq!(from_points.len(), to_points.len());
        debug_assert!(from_points.len() >= 3);

        let mut p = new_col_vector(3, from_points.len());
        let mut q = new_col_vector(2, from_points.len());

        for i in 0 .. from_points.len() {
            p[(0, i)] = from_points[i].x;
            p[(1, i)] = from_points[i].y;
            p[(2, i)] = 1.0;

            q[(0, i)] = to_points[i].x;
            q[(1, i)] = to_points[i].y;
        }

        let m: Matrix2x3<f32> = q * p.pseudo_inverse(0.0);

        let slice = m.slice((0, 0), (2, 2));
        let data = slice.data.as_slice();
        let mat = Matrix2::new(data[0], data[1], data[2], data[3]);

        let col = m.column(2).into_owned();

        Self::new(mat, col)
    }

    pub fn find_similarity(from_points: &[Vector2], to_points: &[Vector2]) -> Self {
        debug_assert_eq!(from_points.len(), to_points.len());

        let mut mean_to = Vector2::new(0.0, 0.0);
        let mut mean_from = Vector2::new(0.0, 0.0);
        let mut sigma_from = 0.0;

        let mut cov = Matrix2::new(0.0, 0.0, 0.0, 0.0);

        for i in 0 .. from_points.len() {
            mean_from += from_points[i];
            mean_to += to_points[i];
        }

        mean_from /= from_points.len() as f32;
        mean_to /= to_points.len() as f32;

        for i in 0 .. from_points.len() {
            sigma_from += length_squared(from_points[i] - mean_from);
            cov += (to_points[i] - mean_to) * (from_points[i] - mean_from).transpose();
        }

        sigma_from /= from_points.len() as f32;
        cov /= from_points.len() as f32;

        let d = diagm(cov);
        let svd = cov.svd(true, true);

        let u = svd.u.unwrap();
        let v = svd.v_t.unwrap();

        let mut s = Matrix2::identity();

        if cov.determinant() < 0.0 || (cov.determinant() == 0.0 && u.determinant() * v.determinant() < 0.0) {
            if d[(1,1)] < d[(0,0)] {
                s[(1,1)] = -1.0;
            } else {
                s[(0,0)] = -1.0;
            }
        }

        let r = u * s * v.transpose();

        let c = if sigma_from == 0.0 {
            1.0
        } else {
            1.0 / sigma_from * (d * s).trace()
        };

        let t = mean_to - c * r * mean_from;

        Self::new(c * r, t)
    }
}

fn diagm(mut mat: Matrix2<f32>) -> Matrix2<f32> {
    for r in 0 .. mat.nrows() {
        for c in 0 .. mat.ncols() {
            if r != c {
                mat[(c, r)] = 0.0;
            }
        }
    }

    mat
}