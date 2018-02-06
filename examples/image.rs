extern crate shape_predictor;
extern crate image;
extern crate rustface;
extern crate line_drawing;

mod util;

use image::GenericImage;

fn main() {
    let mut detector = util::detector();

    let filename = std::env::args().nth(1).unwrap();
    let img = image::open(&filename).unwrap();

    let predictor = shape_predictor::ShapePredictor::read("examples/face_landmarks.bin").unwrap();

    let (width, height) = img.dimensions();
    let mut data = rustface::ImageData::new(img.to_luma().as_ptr(), width, height);

    let mut rgb = img.to_rgb();

    for face in detector.detect(&mut data) {
        let rect = util::face_to_rect(&face);
        let points = predictor.run(&rgb, &rect);
        util::draw_landmarks(&points, &mut rgb);
    }

    rgb.save("out.png").unwrap();
}