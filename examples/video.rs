extern crate shape_predictor;
extern crate image;
extern crate rustface;
extern crate line_drawing;
extern crate videostream;

mod util;

fn main() {
    let mut detector = util::detector();

    let filename = std::env::args().nth(1).unwrap();
    let mut video = videostream::VideoStream::new(&filename).unwrap();

    let predictor = shape_predictor::ShapePredictor::read("examples/face_landmarks.bin").unwrap();

    for (i, frame) in video.iter().enumerate() {
        println!("{}", i);
        let luma = frame.as_luma().unwrap();
        let (width, height) = luma.dimensions();
        let mut data = rustface::ImageData::new(luma.as_ptr(), width, height);

        let faces = detector.detect(&mut data);

        if !faces.is_empty() {
            let mut rgb = frame.as_rgb().unwrap();

            for face in faces {
                let rect = util::face_to_rect(&face);
                let points = predictor.run(&rgb, &rect);
                util::draw_landmarks(&points, &mut rgb);
            }

            rgb.save(&format!("{}.png", i)).unwrap();
        }
    }   
}