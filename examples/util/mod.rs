use rustface;
use shape_predictor;
use image;
use line_drawing;

pub fn detector() -> Box<rustface::Detector> {
    let mut detector = rustface::create_detector("examples/seeta_fd_frontal_v1.0.bin").unwrap();
    detector.set_min_face_size(20);
    detector.set_score_thresh(2.0);
    detector.set_pyramid_scale_factor(0.8);
    detector.set_slide_window_step(4, 4);
    detector
}

pub fn face_to_rect(face: &rustface::FaceInfo) -> shape_predictor::Rectangle {
    let bbox = face.bbox();
    shape_predictor::Rectangle::new(bbox.x() as f32, bbox.y() as f32, bbox.width() as f32, bbox.height() as f32)
}

pub fn draw_landmarks(landmarks: &[shape_predictor::Vector2], image: &mut image::RgbImage) {
    [
        // Chin
        &landmarks[0 .. 17],
        // Right eyebrow
        &landmarks[17 .. 22],
        // Left eyebrow
        &landmarks[22 .. 27],
        // Nose
        &landmarks[27 .. 31],
        // Bottom part of nose
        &landmarks[31 .. 36],
        &[landmarks[30], landmarks[33]],
        // Right eye
        &landmarks[36 .. 42],
        &[landmarks[36], landmarks[41]],
        // Left eye
        &landmarks[42 .. 48],
        &[landmarks[42], landmarks[47]],
        // Outer lips
        &landmarks[48 .. 60],
        &[landmarks[48], landmarks[59]],
        // Inner lips
        &landmarks[60 .. 68],
        &[landmarks[60], landmarks[67]]
    ].iter()
        .flat_map(|part| part.windows(2))
        .flat_map(|window| line_drawing::Bresenham::new((window[0].x as i32, window[0].y as i32), (window[1].x as i32, window[1].y as i32)))
        .for_each(|(x, y)| image.put_pixel(x as u32, y as u32, image::Rgb([255, 0, 0])));

    landmarks.iter()
            .for_each(|point| image.put_pixel(point.x as u32, point.y as u32, image::Rgb([0, 255, 0])));
}