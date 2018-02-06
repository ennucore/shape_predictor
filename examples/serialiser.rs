extern crate shape_predictor;

fn main() {
    shape_predictor::ShapePredictor::read_from_dlib("examples/shape_predictor_68_face_landmarks.dat")
        .unwrap()
        .write("examples/face_landmarks.bin")
        .unwrap()
}